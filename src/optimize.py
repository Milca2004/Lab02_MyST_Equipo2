"""Optimización con Optuna y walk-forward (solo dentro de TRAIN).

θ*_{i,j} = argmax_θ Calmar(backtest(train_k | S_t = j, activo i, θ))
- "| S_t = j" se implementa con entry_mask (solo se abre en días del régimen j).
- Si θ produce menos de N_MIN operaciones, el objetivo vale −1e9.
- Sampler: TPESampler(n_startup_trials=30): 30 trials aleatorios y luego TPE.
- Se usa el θ robusto (mediana del 10% de mejores trials válidos), no el argmax.
"""
import logging
import math
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd

from src import config
from src.backtest import datos_un_activo, run_backtest
from src.data import asset_frame
from src.metrics import calmar
from src.regimes import ajustar_modelo, clasificar
from src import signals

# ----------------------------------------------------------------------------
# Espacio de búsqueda θ (cada rango con su justificación)
# ----------------------------------------------------------------------------
ESPACIO = {
    # (tipo, mínimo, máximo)
    "ema_fast": ("int", 5, 30),       # 1 semana a 1.5 meses: tendencia de corto plazo
    "ema_gap": ("int", 10, 100),      # lenta = rápida + gap -> siempre lenta > rápida
    "rsi_window": ("int", 7, 28),     # alrededor del 14 clásico de Wilder (½× a 2×)
    "rsi_lower": ("int", 20, 40),     # sobreventa: de extrema (20) a moderada (40)
    "rsi_upper": ("int", 60, 80),     # sobrecompra: simétrico al umbral inferior
    "bb_window": ("int", 10, 40),     # alrededor del 20 clásico de Bollinger
    "bb_k": ("float", 1.5, 3.0),      # 1.5σ (sensible) a 3σ (solo extremos)
    "m_sl": ("float", 1.0, 4.0),      # SL entre 1 y 4 ATR: ruido diario ≈ 1 ATR
    "m_tp": ("float", 1.0, 6.0),      # TP permite razones ganancia/pérdida > 1
    "max_hold": ("int", 5, 40),       # 1 semana a 2 meses de tenencia máxima
}
PARAMS_ENTEROS = [p for p, (tipo, _, _) in ESPACIO.items() if tipo == "int"]

# θ por defecto (centro del espacio): solo si NINGÚN trial del estudio global
# cumple el mínimo de operaciones (se registra y se reporta).
THETA_DEFECTO = {"ema_fast": 17, "ema_gap": 55, "rsi_window": 17, "rsi_lower": 30,
                 "rsi_upper": 70, "bb_window": 25, "bb_k": 2.25, "m_sl": 2.5,
                 "m_tp": 3.5, "max_hold": 22}


def sugerir_parametros(trial):
    """Pide a Optuna un θ dentro de ESPACIO."""
    params = {}
    for nombre, (tipo, minimo, maximo) in ESPACIO.items():
        if tipo == "int":
            params[nombre] = trial.suggest_int(nombre, minimo, maximo)
        else:
            params[nombre] = trial.suggest_float(nombre, minimo, maximo)
    return params


# ----------------------------------------------------------------------------
# Caché de indicadores (mismas fórmulas de signals.py, calculadas una sola vez)
# ----------------------------------------------------------------------------
class CacheIndicadores:
    """Guarda cada indicador por (indicador, ventana) para no recalcularlo
    en cada trial. Los umbrales (RSI) y el k de Bollinger se aplican después,
    porque no cambian la serie base. El resultado es idéntico a
    signals.generate_signals (hay una prueba que lo verifica)."""

    def __init__(self, ohlcv):
        self.close = ohlcv["close"]
        self.atr = signals.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
        self.emas, self.rsis, self.bandas = {}, {}, {}

    def _ema(self, h):
        if h not in self.emas:
            self.emas[h] = signals.ema(self.close, h)
        return self.emas[h]

    def _rsi(self, n):
        if n not in self.rsis:
            self.rsis[n] = signals.rsi(self.close, n)
        return self.rsis[n]

    def _bandas(self, n):
        if n not in self.bandas:
            self.bandas[n] = (self.close.rolling(n).mean(), self.close.rolling(n).std(ddof=0))
        return self.bandas[n]

    def senales(self, params, umbral=config.UMBRAL_CONFIRMACION, indicadores=None):
        rapida = params["ema_fast"]
        lenta = rapida + params["ema_gap"]
        voto_ema = np.sign(self._ema(rapida) - self._ema(lenta))
        voto_ema.iloc[:lenta] = 0.0

        valor_rsi = self._rsi(params["rsi_window"])
        voto_rsi = pd.Series(0.0, index=self.close.index)
        voto_rsi[valor_rsi < params["rsi_lower"]] = 1.0
        voto_rsi[valor_rsi > params["rsi_upper"]] = -1.0

        media, sigma = self._bandas(params["bb_window"])
        voto_bb = pd.Series(0.0, index=self.close.index)
        voto_bb[self.close < media - params["bb_k"] * sigma] = 1.0
        voto_bb[self.close > media + params["bb_k"] * sigma] = -1.0

        votos = pd.DataFrame({"voto_ema": voto_ema, "voto_rsi": voto_rsi, "voto_bb": voto_bb})
        if indicadores is None:
            indicadores = signals.NOMBRES_VOTOS
        salida = votos.copy()
        salida["suma_votos"] = votos[indicadores].sum(axis=1)
        salida["s"] = signals.confirmar(votos[indicadores], umbral)
        salida["direccion"] = np.sign(salida["s"])
        salida["atr"] = self.atr
        return salida


# ----------------------------------------------------------------------------
# Ventanas del walk-forward
# ----------------------------------------------------------------------------
def bloques_mensuales(fechas, n_train):
    """Bloques mensuales de índices. El mes que cruza el corte TRAIN/TEST se
    parte en dos bloques para que ningún bloque mezcle TRAIN y TEST."""
    etiquetas = [f"{f.year}-{f.month:02d}" + ("T" if i >= n_train else "")
                 for i, f in enumerate(fechas)]
    bloques, actual = [], [0]
    for i in range(1, len(fechas)):
        if etiquetas[i] == etiquetas[i - 1]:
            actual.append(i)
        else:
            bloques.append(np.array(actual))
            actual = [i]
    bloques.append(np.array(actual))
    return bloques


def generar_ventanas(fechas, n_train, hasta_idx, anchored=False):
    """Ventanas del walk-forward: train de 6 bloques mensuales -> 1 mes OOS.

    - El primer train empieza después de DIAS_CALENTAMIENTO (historia para
      calentar indicadores y features; usar datos pasados es válido).
    - Solo se usan bloques con índice ≤ hasta_idx.
    - La última ventana termina justo en hasta_idx y no tiene OOS dentro del
      rango: su θ es el que se congela para el siguiente tramo.
    - anchored=True: el train empieza siempre en el primer bloque (creciente).
    """
    bloques = [b for b in bloques_mensuales(fechas, n_train)
               if b[0] >= config.DIAS_CALENTAMIENTO and b[-1] <= hasta_idx]
    # el primer bloque debe estar completo (empieza en un mes nuevo)
    ventanas = []
    m = config.MESES_TRAIN_WF
    for k in range(len(bloques) - m + 1):
        inicio_bloque = 0 if anchored else k
        train_idx = np.concatenate(bloques[inicio_bloque:k + m])
        oos_idx = bloques[k + m] if k + m < len(bloques) else None
        ventanas.append({
            "k": k,
            "train_ini": fechas[train_idx[0]], "train_fin": fechas[train_idx[-1]],
            "n_meses_train": k + m - inicio_bloque,
            "oos_ini": None if oos_idx is None else fechas[oos_idx[0]],
            "oos_fin": None if oos_idx is None else fechas[oos_idx[-1]],
        })
    return ventanas


# ----------------------------------------------------------------------------
# Estudios de Optuna
# ----------------------------------------------------------------------------
def semilla_estudio(k, i_activo, regimen):
    """Semilla determinística por (ventana, activo, régimen) derivada de SEED.
    i_activo = −1 para la variante compartida; regimen = −1 para el global."""
    return config.SEED * 1_000_000 + k * 1_000 + (i_activo + 1) * 10 + (regimen + 1)


def correr_estudio(evaluar, n_trials, seed, n_min):
    """Estudio de Optuna que maximiza el Calmar con mínimo de operaciones.

    evaluar(params) -> (calmar, n_operaciones)
    """
    def objetivo(trial):
        params = sugerir_parametros(trial)
        valor, n_ops = evaluar(params)
        trial.set_user_attr("n_operaciones", int(n_ops))
        if n_ops < n_min:
            return config.OBJETIVO_INVALIDO
        return valor

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=config.N_STARTUP_TRIALS)
    estudio = optuna.create_study(direction="maximize", sampler=sampler)
    estudio.optimize(objetivo, n_trials=n_trials)
    return estudio


def theta_robusto(estudio):
    """Centro de la mejor meseta: mediana de cada parámetro entre el 10% de
    mejores trials VÁLIDOS (al menos 1), con enteros redondeados.
    Regresa (theta_robusto, theta_argmax, valor_argmax, n_validos);
    theta_robusto = None si ningún trial es válido."""
    validos = [t for t in estudio.trials
               if t.value is not None and t.value > config.OBJETIVO_INVALIDO]
    if not validos:
        return None, None, config.OBJETIVO_INVALIDO, 0
    validos.sort(key=lambda t: t.value, reverse=True)
    n_top = max(1, math.ceil(config.FRACCION_ROBUSTA * len(validos)))
    mejores = validos[:n_top]
    theta = {}
    for nombre in ESPACIO:
        mediana = float(np.median([t.params[nombre] for t in mejores]))
        theta[nombre] = int(round(mediana)) if nombre in PARAMS_ENTEROS else mediana
    return theta, dict(validos[0].params), float(validos[0].value), len(validos)


# ----------------------------------------------------------------------------
# Evaluadores (backtest de un activo = mismo motor con n = 1)
# ----------------------------------------------------------------------------
def backtest_activo(cache, ohlcv_muestra, params, mascara=None, nombre="activo",
                    umbral=config.UMBRAL_CONFIRMACION, indicadores=None):
    """Backtest de un activo en la muestra; cierra todo el último día (purga)."""
    sen = cache.senales(params, umbral, indicadores).loc[ohlcv_muestra.index]
    datos = datos_un_activo(ohlcv_muestra, sen, params, mascara, cerrar_al_final=True,
                            nombre=nombre)
    return run_backtest(datos)


def evaluador_activo(cache, ohlcv_muestra, mascara, nombre):
    def evaluar(params):
        res = backtest_activo(cache, ohlcv_muestra, params, mascara, nombre)
        return calmar(res.equity), res.n_operaciones
    return evaluar


def evaluador_compartido(caches, muestras, mascara):
    """Variante compartida: Calmar de la curva promedio de los 6 backtests
    individuales (cada uno normalizado a 1) y suma de operaciones."""
    def evaluar(params):
        curvas, n_ops = [], 0
        for nombre in caches:
            res = backtest_activo(caches[nombre], muestras[nombre], params, mascara, nombre)
            curvas.append(res.equity / res.equity.iloc[0])
            n_ops += res.n_operaciones
        promedio = pd.concat(curvas, axis=1).mean(axis=1)
        return calmar(promedio), n_ops
    return evaluar


def resumen_estudio(estudio, theta_global=None, motivo_fallback=None):
    """Resultado de un estudio con fallback declarado a θ global."""
    theta, theta_max, valor_max, n_validos = theta_robusto(estudio) if estudio else (None, None, None, 0)
    fallback = theta is None
    if fallback:
        theta = theta_global if theta_global is not None else dict(THETA_DEFECTO)
        motivo_fallback = motivo_fallback or "ningún trial cumple N_MIN"
    return {"theta": theta, "theta_argmax": theta_max, "valor_argmax": valor_max,
            "n_validos": n_validos, "fallback": fallback,
            "motivo_fallback": motivo_fallback if fallback else None,
            "n_trials": len(estudio.trials) if estudio else 0}


# ----------------------------------------------------------------------------
# Una ventana del walk-forward
# ----------------------------------------------------------------------------
def preparar_muestra(panel, features, ventana):
    """Modelo de régimen (≤ train_fin), muestra de train con embargo y etiquetas."""
    fechas = panel["close"].index
    en_train = (fechas >= ventana["train_ini"]) & (fechas <= ventana["train_fin"])
    idx_train = fechas[en_train]
    muestra = idx_train[:-config.DIAS_EMBARGO]       # embargo: fuera del objetivo
    modelo = ajustar_modelo(features, ventana["train_fin"])
    regimen = clasificar(features.loc[:ventana["train_fin"]],
                         [(ventana["train_ini"], modelo)]).loc[muestra]
    return muestra, modelo, regimen


def n_min_escalado(n_min, dias, dias_semestre=126):
    """N_MIN por semestre escalado a la longitud de la muestra: max(n_min, n_min·días/126)."""
    return max(n_min, int(round(n_min * dias / dias_semestre)))


def optimizar_ventana(panel, features, ventana, n_trials, variantes, archivo=None):
    """Optimiza θ (global + 3 regímenes) para las variantes pedidas.

    variantes: subconjunto de ["por_activo", "compartido"].
    Guarda un checkpoint en `archivo` (si se da) y lo reutiliza si existe.
    """
    if archivo is not None and os.path.exists(archivo):
        return joblib.load(archivo)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inicio_reloj = time.time()
    k = ventana["k"]
    muestra, modelo, regimen = preparar_muestra(panel, features, ventana)

    caches, muestras = {}, {}
    for nombre in config.TICKERS:
        ohlcv = asset_frame(panel, nombre).loc[:ventana["train_fin"]]
        caches[nombre] = CacheIndicadores(ohlcv)
        muestras[nombre] = ohlcv.loc[muestra]

    dias_regimen = {j: int((regimen == j).sum()) for j in range(config.N_REGIMENES)}
    # N_MIN está definido por semestre (126 días). En rolling vale exactamente 5 y 3;
    # en anchored crece con los días de la muestra (global) o del régimen (por régimen).
    n_min_global = n_min_escalado(config.N_MIN_GLOBAL, len(muestra))
    n_min_regimen = {j: n_min_escalado(config.N_MIN_REGIMEN, dias_regimen[j])
                     for j in range(config.N_REGIMENES)}
    resultado = {"ventana": ventana, "modelo": modelo, "dias_regimen": dias_regimen,
                 "n_min_global": n_min_global, "n_min_regimen": n_min_regimen}
    n_trials_total = 0

    if "por_activo" in variantes:
        resultado["por_activo"] = {}
        for i, nombre in enumerate(config.TICKERS):
            evaluar = evaluador_activo(caches[nombre], muestras[nombre], None, nombre)
            estudio = correr_estudio(evaluar, n_trials, semilla_estudio(k, i, -1), n_min_global)
            global_ = resumen_estudio(estudio)
            n_trials_total += n_trials
            por_regimen = {"global": global_}
            for j in range(config.N_REGIMENES):
                if dias_regimen[j] < config.MIN_DIAS_REGIMEN:
                    por_regimen[j] = resumen_estudio(None, global_["theta"],
                                                     f"régimen con {dias_regimen[j]} días < 20")
                    continue
                mascara = (regimen == j).to_numpy()
                evaluar = evaluador_activo(caches[nombre], muestras[nombre], mascara, nombre)
                estudio = correr_estudio(evaluar, n_trials, semilla_estudio(k, i, j), n_min_regimen[j])
                por_regimen[j] = resumen_estudio(estudio, global_["theta"])
                n_trials_total += n_trials
            resultado["por_activo"][nombre] = por_regimen

    if "compartido" in variantes:
        n_activos = len(config.TICKERS)
        evaluar = evaluador_compartido(caches, muestras, None)
        estudio = correr_estudio(evaluar, n_trials, semilla_estudio(k, -1, -1),
                                 n_activos * n_min_global)
        global_ = resumen_estudio(estudio)
        n_trials_total += n_trials
        por_regimen = {"global": global_}
        for j in range(config.N_REGIMENES):
            if dias_regimen[j] < config.MIN_DIAS_REGIMEN:
                por_regimen[j] = resumen_estudio(None, global_["theta"],
                                                 f"régimen con {dias_regimen[j]} días < 20")
                continue
            mascara = (regimen == j).to_numpy()
            evaluar = evaluador_compartido(caches, muestras, mascara)
            estudio = correr_estudio(evaluar, n_trials, semilla_estudio(k, -1, j),
                                     n_activos * n_min_regimen[j])
            por_regimen[j] = resumen_estudio(estudio, global_["theta"])
            n_trials_total += n_trials
        resultado["compartido"] = por_regimen

    resultado["n_trials_total"] = n_trials_total
    resultado["segundos"] = time.time() - inicio_reloj
    if archivo is not None:
        joblib.dump(resultado, archivo)
    return resultado


def correr_walk_forward(panel, features, ventanas, n_trials, variantes, dir_checkpoint,
                        n_jobs=None):
    """Optimiza todas las ventanas en paralelo (procesos) con checkpoint por ventana.

    Se paraleliza por ventana: las ventanas son independientes (θ_k solo usa
    train_k), así se aprovechan todos los núcleos aunque haya solo 6 activos.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    os.makedirs(dir_checkpoint, exist_ok=True)
    tareas = []
    for v in ventanas:
        archivo = os.path.join(dir_checkpoint, f"ventana_{v['k']:03d}.pkl")
        tareas.append(joblib.delayed(optimizar_ventana)(panel, features, v, n_trials,
                                                        variantes, archivo))
    logging.info("Walk-forward: %d ventanas, %d trials por estudio, %d procesos",
                 len(ventanas), n_trials, n_jobs)
    return joblib.Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(tareas)


# ----------------------------------------------------------------------------
# Utilidades sobre los resultados
# ----------------------------------------------------------------------------
def aplicar_fallback_previo(resultados, variante):
    """Fallback global causal: si en la ventana k ningún trial del estudio global
    fue válido, se usa el θ global de la última ventana previa que sí lo tuvo
    (el sistema conserva los últimos parámetros optimizados). Los regímenes en
    fallback de esa ventana heredan ese mismo θ. Solo si no existe ninguna
    ventana previa válida se queda el θ por defecto. Modifica `resultados`."""
    ultimo_valido, k_valido = {}, {}
    for res in resultados:
        bloques = (res[variante].values() if variante == "por_activo" else [res[variante]])
        for i, estudios in enumerate(bloques):
            clave = i if variante == "por_activo" else 0
            global_ = estudios["global"]
            if not global_["fallback"]:
                ultimo_valido[clave] = global_["theta"]
                k_valido[clave] = res["ventana"]["k"]
            elif clave in ultimo_valido:
                global_["theta"] = ultimo_valido[clave]
                global_["motivo_fallback"] = (f"sin trials válidos: θ global de la ventana "
                                              f"{k_valido[clave]}")
            for j in range(config.N_REGIMENES):
                if estudios[j]["fallback"]:
                    estudios[j]["theta"] = global_["theta"]
    return resultados


def thetas_ventana(resultado, variante):
    """{activo: {régimen: θ}} de una ventana para la variante dada."""
    thetas = {}
    for nombre in config.TICKERS:
        if variante == "por_activo":
            estudios = resultado["por_activo"][nombre]
        else:
            estudios = resultado["compartido"]
        thetas[nombre] = {j: estudios[j]["theta"] for j in range(config.N_REGIMENES)}
    return thetas


def tabla_parametros(resultados, variante, esquema):
    """Tabla larga de θ por ventana, activo y régimen (parametros_por_ventana.csv)."""
    filas = []
    for res in resultados:
        v = res["ventana"]
        bloque = res[variante]
        for nombre in config.TICKERS:
            estudios = bloque[nombre] if variante == "por_activo" else bloque
            for clave in ["global"] + list(range(config.N_REGIMENES)):
                e = estudios[clave]
                regimen = "Global" if clave == "global" else config.REGIMENES[clave]
                fila = {"esquema": esquema, "variante": variante, "k": v["k"],
                        "train_ini": v["train_ini"].date(), "train_fin": v["train_fin"].date(),
                        "oos_ini": v["oos_ini"].date() if v["oos_ini"] is not None else None,
                        "activo": nombre if variante == "por_activo" else "TODOS",
                        "regimen": regimen, "fallback": e["fallback"],
                        "motivo_fallback": e["motivo_fallback"],
                        "n_validos": e["n_validos"], "calmar_is_argmax": e["valor_argmax"]}
                fila.update({p: e["theta"][p] for p in ESPACIO})
                filas.append(fila)
            if variante == "compartido":
                break
    return pd.DataFrame(filas)


def estudio_diagnostico(panel, features, ventana, nombre, n_trials):
    """Re-ejecuta (con la misma semilla, resultado idéntico) el estudio global
    de un activo en una ventana, para los diagnósticos de Optuna."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    muestra, _, _ = preparar_muestra(panel, features, ventana)
    ohlcv = asset_frame(panel, nombre).loc[:ventana["train_fin"]]
    cache = CacheIndicadores(ohlcv)
    evaluar = evaluador_activo(cache, ohlcv.loc[muestra], None, nombre)
    i = config.TICKERS.index(nombre)
    estudio = correr_estudio(evaluar, n_trials, semilla_estudio(ventana["k"], i, -1),
                             config.N_MIN_GLOBAL)
    return estudio, evaluar


# ----------------------------------------------------------------------------
# Del walk-forward a la simulación continua
# ----------------------------------------------------------------------------
def horario_theta(resultados, variante):
    """[(oos_ini, oos_fin, thetas)]: el mes k+1 usa lo optimizado en la ventana k."""
    horario = []
    for res in resultados:
        v = res["ventana"]
        if v["oos_ini"] is not None:
            horario.append((v["oos_ini"], v["oos_fin"], thetas_ventana(res, variante)))
    return horario


def horario_modelos(resultados):
    """[(oos_ini, modelo_k)]: el régimen del mes k+1 usa el modelo de la ventana k."""
    return [(res["ventana"]["oos_ini"], res["modelo"]) for res in resultados
            if res["ventana"]["oos_ini"] is not None]


def fechas_oos(resultados, fechas):
    """Fechas del tramo WF-OOS concatenado."""
    con_oos = [r["ventana"] for r in resultados if r["ventana"]["oos_ini"] is not None]
    return fechas[(fechas >= con_oos[0]["oos_ini"]) & (fechas <= con_oos[-1]["oos_fin"])]


def simular_wf_is(contexto_base, features, resultados, variante, frecuencia, delta):
    """WF-IS: portafolio de cada ventana sobre su propio train (con θ_k y el
    régimen in-sample del modelo k), con purga al final. Una fila por ventana."""
    from src.metrics import cagr, calmar, max_drawdown
    from src.portfolio import ContextoSimulacion, construir_senales

    panel = contexto_base.panel
    filas = []
    for res in resultados:
        v = res["ventana"]
        muestra, modelo, regimen = preparar_muestra(panel, features, v)
        thetas = thetas_ventana(res, variante)
        senales = construir_senales(panel, regimen, [(muestra[0], muestra[-1], thetas)], muestra)
        contexto = ContextoSimulacion(panel, regimen, muestra, contexto_base.cache_pesos)
        sim = contexto.simular(senales, "rp", frecuencia, delta, cerrar_al_final=True)
        filas.append({"k": v["k"], "train_ini": v["train_ini"].date(),
                      "train_fin": v["train_fin"].date(), "variante": variante,
                      "cagr_is": cagr(sim.equity), "calmar_is": calmar(sim.equity),
                      "mdd_is": max_drawdown(sim.equity), "n_operaciones_is": sim.n_operaciones})
    return pd.DataFrame(filas)


def walk_forward_efficiency(cagr_oos, calmar_oos, tabla_is):
    """WFE = rendimiento anualizado WF-OOS / rendimiento anualizado WF-IS promedio.
    También en versión Calmar. Por debajo de ~0.5, el in-sample es mayormente ruido."""
    cagr_is = float(tabla_is["cagr_is"].mean())
    calmar_is = float(tabla_is["calmar_is"].mean())
    return {"cagr_is_promedio": cagr_is, "cagr_oos": cagr_oos,
            "wfe_rendimiento": cagr_oos / cagr_is if cagr_is != 0 else np.nan,
            "calmar_is_promedio": calmar_is, "calmar_oos": calmar_oos,
            "wfe_calmar": calmar_oos / calmar_is if calmar_is != 0 else np.nan}


# ----------------------------------------------------------------------------
# Robustez (sección 13)
# ----------------------------------------------------------------------------
def fila_metricas(etiqueta, res):
    from src.metrics import cagr, calmar, max_drawdown, sharpe
    return {"caso": etiqueta, "calmar": calmar(res.equity), "sharpe": sharpe(res.equity),
            "mdd": max_drawdown(res.equity), "cagr": cagr(res.equity),
            "n_operaciones": res.n_operaciones, "costos_totales": res.costos_totales}


def escalar_theta(theta, parametro, factor):
    """θ con un parámetro multiplicado por (1 + factor); enteros redondeados, mínimo 1."""
    nuevo = dict(theta)
    valor = theta[parametro] * (1.0 + factor)
    if parametro in PARAMS_ENTEROS:
        valor = max(1, int(round(valor)))
    nuevo[parametro] = valor
    return nuevo


def sensibilidad(contexto, thetas, frecuencia, delta, factores=(-0.2, -0.1, 0.0, 0.1, 0.2)):
    """±20% uno a la vez (a todos los activos y regímenes) sobre θ, m(régimen),
    δ y la ventana de Σ, con el θ congelado como base."""
    from src.portfolio import construir_senales

    fechas = contexto.fechas
    filas = []
    for parametro in ESPACIO:
        for factor in factores:
            modificado = {a: {j: escalar_theta(thetas[a][j], parametro, factor)
                              for j in thetas[a]} for a in thetas}
            senales = construir_senales(contexto.panel, contexto.regimen,
                                        [(fechas[0], fechas[-1], modificado)], fechas)
            res = contexto.simular(senales, "rp", frecuencia, delta)
            fila = fila_metricas(parametro, res)
            fila.update({"parametro": parametro, "factor": factor})
            filas.append(fila)
    base = construir_senales(contexto.panel, contexto.regimen,
                             [(fechas[0], fechas[-1], thetas)], fechas)
    for factor in factores:
        m = {j: min(1.0, valor * (1.0 + factor)) for j, valor in config.M_REGIMEN.items()}
        casos = {
            "m_regimen": contexto.simular(base, "rp", frecuencia, delta, m_regimen=m),
            "delta": contexto.simular(base, "rp", frecuencia, delta * (1.0 + factor)),
            "ventana_cov": contexto.simular(base, "rp", frecuencia, delta,
                                            ventana=int(round(config.VENTANA_COV * (1.0 + factor)))),
        }
        for parametro, res in casos.items():
            fila = fila_metricas(parametro, res)
            fila.update({"parametro": parametro, "factor": factor})
            filas.append(fila)
    return pd.DataFrame(filas)


def curva_costos(contexto, senales, frecuencia, delta, paso=0.00025, maximo=0.005):
    """Retorno neto vs. comisión (0% a 0.5% en pasos de 0.025%).
    Punto de equilibrio: comisión donde el retorno neto cruza 0 (interpolación lineal).
    Margen de seguridad = equilibrio / 0.125%."""
    from src.backtest import Costos
    from src.portfolio import descomponer_resultado

    filas = []
    for comision in np.arange(0.0, maximo + 1e-12, paso):
        res = contexto.simular(senales, "rp", frecuencia, delta, costos=Costos(comision=comision))
        fila = fila_metricas(f"{comision:.5f}", res)
        fila.update(descomponer_resultado(res))
        fila["comision"] = comision
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    equilibrio = punto_equilibrio(tabla["comision"].to_numpy(), tabla["retorno_neto"].to_numpy())
    return tabla, equilibrio


def punto_equilibrio(x, y):
    """Primer cruce de y por cero (interpolación lineal); NaN si no cruza."""
    for k in range(1, len(x)):
        if y[k - 1] > 0 >= y[k]:
            return float(x[k - 1] + (x[k] - x[k - 1]) * y[k - 1] / (y[k - 1] - y[k]))
    return float("nan") if y[0] > 0 else 0.0


def escenarios_ejecucion(contexto, senales, frecuencia, delta):
    """Costos realistas: oficial vs. +spread 2 bps, +borrow 0.5% anual, +impacto
    η·|q/ADV|^(2/3) y los tres juntos; además barrido de slippage 0–20 bps."""
    from src.backtest import Costos

    escenarios = {
        "oficial (solo comisión)": Costos(),
        "+ spread 2 bps": Costos(spread_bps=config.SPREAD_BPS_REALISTA),
        "+ borrow fee 0.5%": Costos(borrow_fee_anual=config.BORROW_FEE_REALISTA),
        "+ impacto η=0.1": Costos(impacto_eta=config.IMPACTO_ETA),
        "realista (todo junto)": Costos(spread_bps=config.SPREAD_BPS_REALISTA,
                                        borrow_fee_anual=config.BORROW_FEE_REALISTA,
                                        impacto_eta=config.IMPACTO_ETA),
    }
    filas = [fila_metricas(nombre, contexto.simular(senales, "rp", frecuencia, delta, costos=c))
             for nombre, c in escenarios.items()]
    slippage = []
    for bps in np.arange(0.0, 20.0 + 1e-9, 2.5):
        res = contexto.simular(senales, "rp", frecuencia, delta, costos=Costos(slippage_bps=bps))
        fila = fila_metricas(f"{bps:.1f} bps", res)
        fila["slippage_bps"] = bps
        slippage.append(fila)
    return pd.DataFrame(filas), pd.DataFrame(slippage)


def experimento_un_indicador(contexto, horario, frecuencia, delta):
    """2 de 3 vs. cada indicador solo (umbral 1, k = 1), con el mismo θ por ventana.
    Se reporta a nivel portafolio y por activo (estrategia individual)."""
    from src.metrics import calmar
    from src.portfolio import construir_senales, simular_activo_individual

    casos = {"2 de 3": (None, config.UMBRAL_CONFIRMACION),
             "solo EMA": (["voto_ema"], 1), "solo RSI": (["voto_rsi"], 1),
             "solo Bollinger": (["voto_bb"], 1)}
    filas = []
    for etiqueta, (indicadores, umbral) in casos.items():
        senales = construir_senales(contexto.panel, contexto.regimen, horario, contexto.fechas,
                                    umbral, indicadores)
        res = contexto.simular(senales, "rp", frecuencia, delta)
        fila = fila_metricas(etiqueta, res)
        fila["nivel"] = "portafolio"
        filas.append(fila)
        for i, activo in enumerate(config.TICKERS):
            ind = simular_activo_individual(contexto.panel, senales, i, contexto.fechas)
            filas.append({"caso": etiqueta, "nivel": activo, "calmar": calmar(ind.equity),
                          "n_operaciones": ind.n_operaciones})
    return pd.DataFrame(filas)
