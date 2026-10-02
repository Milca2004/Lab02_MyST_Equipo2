"""Pipeline completo del Lab 02 (MyST, ITESO) — Equipo 2, Nivel C.

Uso:
    python main.py                      # todo: data -> train -> test -> report
    python main.py --stage train        # solo una etapa
    python main.py --quick              # ensayo rápido (20 trials, pocas ventanas)
                                        # escribe en .cache/quick/, nunca en docs/

main.py solo orquesta: toda la lógica vive en src/.
"""
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src import config, data, metrics, optimize, portfolio, regimes, signals

VARIANTES = ["por_activo", "compartido"]
VENTANAS_QUICK = 4
TRIALS_QUICK = 20


# ----------------------------------------------------------------------------
# Utilidades de entrada/salida
# ----------------------------------------------------------------------------
class Salidas:
    """Rutas de salida y funciones para guardar resultados."""

    def __init__(self, quick):
        self.quick = quick
        self.resultados, self.figuras, self.cache = config.rutas_salida(quick)

    def csv(self, tabla, nombre, index=False):
        tabla.to_csv(self.resultados / nombre, index=index)

    def json(self, objeto, nombre):
        with open(self.resultados / nombre, "w", encoding="utf-8") as f:
            json.dump(objeto, f, indent=2, ensure_ascii=False, default=_a_json)

    def leer_json(self, nombre):
        with open(self.resultados / nombre, encoding="utf-8") as f:
            return json.load(f)


def _a_json(valor):
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        return float(valor)
    if isinstance(valor, (np.bool_,)):
        return bool(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    return str(valor)


def sha256_archivo(ruta):
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def truncar(panel, n):
    """Primeras n filas del panel (nada posterior entra a los cálculos)."""
    return {campo: tabla.iloc[:n] for campo, tabla in panel.items()}


def arbol_limpio():
    """True solo si hay repo con al menos un commit y git status está vacío."""
    codigo_estado, estado = git("status", "--porcelain")
    codigo_head, _ = git("rev-parse", "HEAD")
    return codigo_estado == 0 and codigo_head == 0 and estado == ""


def git(*args):
    resultado = subprocess.run(["git", *args], capture_output=True, text=True,
                               cwd=config.RAIZ)
    return resultado.returncode, resultado.stdout.strip()


# ----------------------------------------------------------------------------
# Etapa DATA
# ----------------------------------------------------------------------------
def etapa_data(salidas):
    if data.ensure_data():
        logging.info("Datos descargados y congelados en %s", config.ARCHIVO_PRECIOS)
    else:
        logging.info("Usando datos congelados (no se descarga): %s", config.ARCHIVO_PRECIOS)
    largo = data.load_long()
    salidas.csv(data.audit_data(largo), "auditoria_datos.csv")
    panel, eliminadas = data.clean_data(data.to_panel(largo))
    fechas = panel["close"].index
    split = data.split_train_test(fechas)
    split["filas_eliminadas_limpieza"] = eliminadas
    split["n_total"] = len(fechas)
    if salidas.quick:
        # Ensayo: el "TEST" son los últimos 2 meses de TRAIN; el TEST real no se usa.
        bloques = [b for b in optimize.bloques_mensuales(fechas, split["n_train"])
                   if b[-1] < split["n_train"]]
        inicio_test = int(bloques[-2][0])
        split = {"train_inicio": split["train_inicio"],
                 "train_fin": str(fechas[inicio_test - 1].date()),
                 "test_inicio": str(fechas[inicio_test].date()),
                 "test_fin": str(fechas[split["n_train"] - 1].date()),
                 "n_train": inicio_test, "n_test": split["n_train"] - inicio_test,
                 "n_total": split["n_train"], "filas_eliminadas_limpieza": eliminadas,
                 "nota": "ENSAYO --quick: TEST = últimos 2 meses de TRAIN"}
    salidas.json(split, "splits.json")
    logging.info("Split: TRAIN %s a %s | TEST %s a %s", split["train_inicio"],
                 split["train_fin"], split["test_inicio"], split["test_fin"])
    return panel, split


# ----------------------------------------------------------------------------
# Etapa TRAIN
# ----------------------------------------------------------------------------
def etapa_train(panel_completo, split, salidas):
    reloj = time.time()
    n_train = split["n_train"]
    panel = truncar(panel_completo, n_train)          # solo TRAIN
    fechas = panel["close"].index
    features = regimes.calcular_features(panel)
    n_trials = TRIALS_QUICK if salidas.quick else config.N_TRIALS

    # ---- walk-forward rolling (ambas variantes) --------------------------
    ventanas = optimize.generar_ventanas(fechas, n_train, n_train - 1)
    ventanas_anc = optimize.generar_ventanas(fechas, n_train, n_train - 1, anchored=True)
    if salidas.quick:
        ventanas, ventanas_anc = ventanas[-VENTANAS_QUICK:], ventanas_anc[-VENTANAS_QUICK:]
    reloj_opt = time.time()
    res_roll = optimize.correr_walk_forward(panel, features, ventanas, n_trials, VARIANTES,
                                            salidas.cache / f"wf_rolling_t{n_trials}")
    seg_rolling = sum(r["segundos"] for r in res_roll)
    for variante in VARIANTES:
        optimize.aplicar_fallback_previo(res_roll, variante)
    logging.info("WF rolling listo (%d ventanas, %.0f s de CPU)", len(res_roll), seg_rolling)

    regimen = regimes.clasificar(features, optimize.horario_modelos(res_roll))
    f_oos = optimize.fechas_oos(res_roll, fechas)
    ctx = portfolio.ContextoSimulacion(panel, regimen, f_oos)

    # ---- elección de variante por Calmar WF-OOS ---------------------------
    filas_var, senales_var, tablas_is = [], {}, []
    for variante in VARIANTES:
        horario = optimize.horario_theta(res_roll, variante)
        senales = portfolio.construir_senales(panel, regimen, horario, f_oos)
        sim = ctx.simular(senales)
        senales_var[variante] = (horario, senales, sim)
        fila = metrics.summary(sim.equity, sim, f"rolling | {variante}")
        tabla_is = optimize.simular_wf_is(ctx, features, res_roll, variante,
                                          config.FRECUENCIA_DEFAULT, config.DELTA_DEFAULT)
        tablas_is.append(tabla_is.assign(esquema="rolling"))
        fila.update(optimize.walk_forward_efficiency(fila["cagr"], fila["calmar"], tabla_is))
        fila.update({"esquema": "rolling", "variante": variante})
        filas_var.append(fila)
    elegida = max(filas_var, key=lambda f: f["calmar"])["variante"]
    logging.info("Variante elegida por Calmar WF-OOS: %s", elegida)
    horario, senales, _ = senales_var[elegida]

    # ---- anchored con la variante elegida ---------------------------------
    res_anc = optimize.correr_walk_forward(panel, features, ventanas_anc, n_trials, [elegida],
                                           salidas.cache / f"wf_anchored_t{n_trials}")
    seg_anchored = sum(r["segundos"] for r in res_anc)
    optimize.aplicar_fallback_previo(res_anc, elegida)
    regimen_anc = regimes.clasificar(features, optimize.horario_modelos(res_anc))
    ctx_anc = portfolio.ContextoSimulacion(panel, regimen_anc, f_oos, ctx.cache_pesos)
    horario_anc = optimize.horario_theta(res_anc, elegida)
    senales_anc = portfolio.construir_senales(panel, regimen_anc, horario_anc, f_oos)
    sim_anc = ctx_anc.simular(senales_anc)
    fila = metrics.summary(sim_anc.equity, sim_anc, f"anchored | {elegida}")
    tabla_is = optimize.simular_wf_is(ctx_anc, features, res_anc, elegida,
                                      config.FRECUENCIA_DEFAULT, config.DELTA_DEFAULT)
    tablas_is.append(tabla_is.assign(esquema="anchored"))
    fila.update(optimize.walk_forward_efficiency(fila["cagr"], fila["calmar"], tabla_is))
    fila.update({"esquema": "anchored", "variante": elegida})
    filas_var.append(fila)
    salidas.csv(pd.DataFrame(filas_var), "comparacion_variantes.csv")
    salidas.csv(pd.concat(tablas_is), "wf_is_por_ventana.csv")
    parametros = pd.concat([optimize.tabla_parametros(res_roll, v, "rolling") for v in VARIANTES]
                           + [optimize.tabla_parametros(res_anc, elegida, "anchored")])
    salidas.csv(parametros, "parametros_por_ventana.csv")

    # ---- barrido de rebalanceo (f × δ) -------------------------------------
    barrido = portfolio.barrido_rebalanceo(ctx, senales)
    salidas.csv(barrido, "rebalanceo_barrido.csv")
    mejor = barrido.loc[barrido["calmar"].idxmax()]
    frecuencia, delta = str(mejor["frecuencia"]), float(mejor["delta"])
    logging.info("Rebalanceo elegido en TRAIN: %s, δ = %.2f", frecuencia, delta)

    # ---- simulaciones oficiales WF-OOS ------------------------------------
    sims = {m: ctx.simular(senales, m, frecuencia, delta) for m in ["rp", "naive", "ew"]}
    guardar_conjunto(salidas, "wf_oos", panel, sims, senales, regimen, f_oos)
    curvas = pd.read_csv(salidas.resultados / "equity_wf_oos.csv", index_col=0, parse_dates=True)
    curvas[f"anchored ({elegida})"] = sim_anc.equity
    for variante in VARIANTES:
        curvas[f"rolling {variante} (rebalanceo default)"] = senales_var[variante][2].equity
    curvas.to_csv(salidas.resultados / "equity_wf_oos.csv")

    # ---- θ congelado ------------------------------------------------------
    ultimo = res_roll[-1]
    thetas_final = optimize.thetas_ventana(ultimo, elegida)
    ruta_modelo = salidas.resultados / "modelo_regimen_congelado.pkl"
    joblib.dump(ultimo["modelo"], ruta_modelo)
    congelado = {
        "variante": elegida, "ventana_k": ultimo["ventana"]["k"],
        "train_ini_ultima_ventana": ultimo["ventana"]["train_ini"].date(),
        "train_fin_ultima_ventana": ultimo["ventana"]["train_fin"].date(),
        "thetas": {a: {config.REGIMENES[j]: t for j, t in thetas_final[a].items()}
                   for a in thetas_final},
        "m_regimen": {config.REGIMENES[j]: m for j, m in config.M_REGIMEN.items()},
        "rebalanceo": {"frecuencia": frecuencia, "delta": delta},
        "metodo_pesos": "rp (Spinu)", "estimador_sigma": "muestral",
        "ventana_cov": config.VENTANA_COV,
        "modelo_regimen": {"archivo": ruta_modelo.name, "sha256": sha256_archivo(ruta_modelo),
                           "features": ultimo["modelo"].features,
                           "mapeo": {str(k): config.REGIMENES[v]
                                     for k, v in ultimo["modelo"].mapeo.items()}},
        "split": split,
    }
    salidas.json(congelado, "theta_congelado.json")
    hash_theta = sha256_archivo(salidas.resultados / "theta_congelado.json")
    logging.info("θ congelado SHA-256: %s", hash_theta)

    # ---- robustez ---------------------------------------------------------
    regimen_cong = regimes.clasificar(features, [(f_oos[0], ultimo["modelo"])])
    ctx_cong = portfolio.ContextoSimulacion(panel, regimen_cong, f_oos, ctx.cache_pesos)
    salidas.csv(optimize.sensibilidad(ctx_cong, thetas_final, frecuencia, delta), "sensibilidad.csv")
    curva, equilibrio = optimize.curva_costos(ctx, senales, frecuencia, delta)
    salidas.csv(curva, "curva_costos.csv")
    senales_cong = portfolio.construir_senales(panel, regimen_cong,
                                               [(f_oos[0], f_oos[-1], thetas_final)], f_oos)
    escenarios, slippage = optimize.escenarios_ejecucion(ctx_cong, senales_cong, frecuencia, delta)
    salidas.csv(escenarios, "costos_realistas_train.csv")
    salidas.csv(slippage, "slippage_train.csv")
    salidas.csv(optimize.experimento_un_indicador(ctx, horario, frecuencia, delta),
                "un_indicador.csv")
    resultado_rp = sims["rp"]
    anios = len(f_oos) / config.DIAS_ANIO
    ops_anio = resultado_rp.n_operaciones / anios
    costo_op = resultado_rp.costos_totales / max(1, 2 * resultado_rp.n_operaciones)
    salidas.json({"comision_equilibrio": equilibrio,
                  "margen_seguridad": equilibrio / config.COMISION if np.isfinite(equilibrio) else None,
                  "operaciones_por_anio": ops_anio,
                  "costo_promedio_por_lado": costo_op,
                  "costo_anual_por_turnover": resultado_rp.costos_totales / anios,
                  "costo_anual_pct_capital": resultado_rp.costos_totales / anios / config.CAPITAL_INICIAL},
                 "costos_resumen.json")

    # ---- Risk Parity: contribuciones, ilusión 50/50, estimadores ------------
    rend = panel["close"].pct_change()
    contrib = portfolio.contribuciones_riesgo(rend, f_oos)
    salidas.csv(contrib, "contribuciones_riesgo.csv")
    promedio = contrib.groupby(["metodo", "activo"])[["peso", "rc_pct"]].mean().reset_index()
    salidas.csv(promedio, "contribuciones_riesgo_promedio.csv")
    estimadores, series_pesos = portfolio.comparar_estimadores(rend, f_oos, frecuencia)
    salidas.csv(estimadores, "estimadores_sigma.csv")
    pd.concat(series_pesos, names=["estimador"]).to_csv(salidas.resultados / "pesos_estimadores.csv")
    pesos_rp, _, condicion = ctx.pesos("rp", frecuencia)
    salidas.csv(pd.DataFrame({"numero_condicion": condicion.loc[f_oos]}), "numero_condicion_sigma.csv",
                index=True)
    filas_rp = []
    for metodo, sim in sims.items():
        fila = metrics.summary(sim.equity, sim, metodo)
        rc = promedio[promedio["metodo"] == metodo].set_index("activo")["rc_pct"]
        fila["rc_max"] = float(rc.max())
        fila["rc_tsla_nvda"] = float(rc[["TSLA", "NVDA"]].sum())
        fila["peso_tsla_nvda"] = float(promedio[promedio["metodo"] == metodo]
                                       .set_index("activo").loc[["TSLA", "NVDA"], "peso"].sum())
        filas_rp.append(fila)
    salidas.csv(pd.DataFrame(filas_rp), "rp_vs_naive_vs_ew.csv")

    # ---- régimen: validación ------------------------------------------------
    validar_regimenes(salidas, panel, features, res_roll, regimen, f_oos, sims["rp"],
                      "wf_oos", ultimo["modelo"])

    # ---- señales: correlación entre votos (θ congelado, régimen Tendencia) --
    filas_corr = []
    for activo in config.TICKERS:
        ohlcv = data.asset_frame(panel, activo)
        votos = signals.generate_signals(ohlcv, thetas_final[activo][config.TENDENCIA])
        corr = signals.correlacion_votos(votos.loc[f_oos])
        for a in corr.index:
            for b in corr.columns:
                filas_corr.append({"activo": activo, "voto_a": a, "voto_b": b,
                                   "correlacion": corr.loc[a, b]})
    salidas.csv(pd.DataFrame(filas_corr), "correlacion_senales.csv")

    # ---- diagnósticos de Optuna --------------------------------------------
    diagnosticos_optuna(salidas, panel, features, ultimo, elegida, n_trials)

    # ---- resumen de la optimización -----------------------------------------
    n_trials_total = sum(r["n_trials_total"] for r in res_roll + res_anc)
    fallbacks = parametros[parametros["regimen"] != "Global"]
    salidas.json({
        "n_trials_por_estudio": n_trials,
        "ventanas_rolling": len(res_roll), "ventanas_anchored": len(res_anc),
        "trials_totales_train": n_trials_total,
        "configuraciones_evaluadas_train": n_trials_total,
        "segundos_cpu_optimizacion_rolling": seg_rolling,
        "segundos_cpu_optimizacion_anchored": seg_anchored,
        "segundos_reloj_etapa_train": time.time() - reloj,
        "segundos_reloj_optimizacion": time.time() - reloj_opt,
        "variante_elegida": elegida,
        "rebalanceo_elegido": {"frecuencia": frecuencia, "delta": delta},
        "fallbacks_por_regimen": fallbacks.groupby(["esquema", "variante", "regimen"])["fallback"]
                                          .agg(["sum", "count"]).reset_index().to_dict("records"),
        "fallbacks_globales": fallbacks_globales(parametros),
        "n_procesos": max(1, (os.cpu_count() or 2) - 1),
        "n_min_global": config.N_MIN_GLOBAL, "n_min_regimen": config.N_MIN_REGIMEN,
    }, "optimizacion_resumen.json")
    logging.info("Etapa TRAIN terminada en %.1f min", (time.time() - reloj) / 60)


def fallbacks_globales(parametros):
    """Estudios globales sin trials válidos: cuántos heredaron el θ de una ventana
    previa y cuántos usaron el θ por defecto (por esquema y variante)."""
    globales = parametros[(parametros["regimen"] == "Global") & parametros["fallback"]]
    filas = []
    for (esquema, variante), g in globales.groupby(["esquema", "variante"]):
        heredados = int(g["motivo_fallback"].str.contains("ventana", na=False).sum())
        total = int((parametros["regimen"] == "Global")[(parametros["esquema"] == esquema)
                                                       & (parametros["variante"] == variante)].sum())
        filas.append({"esquema": esquema, "variante": variante, "estudios": total,
                      "sin_trials_validos": len(g), "heredados_ventana_previa": heredados,
                      "theta_por_defecto": len(g) - heredados})
    return filas


def guardar_conjunto(salidas, conjunto, panel, sims, senales, regimen, fechas):
    """Métricas, curvas, tablas de rendimientos, operaciones y métricas por
    activo y por régimen de un conjunto (wf_oos o test)."""
    bh = portfolio.buy_and_hold(panel, fechas)
    individuales = {a: portfolio.simular_activo_individual(panel, senales, i, fechas)
                    for i, a in enumerate(config.TICKERS)}
    nombres = {"rp": "RP (sistema)", "naive": "RP naive", "ew": "Pesos iguales (EW)"}
    filas = [metrics.summary(sim.equity, sim, nombres[m]) for m, sim in sims.items()]
    filas.append(metrics.summary(bh, None, "Buy & Hold EW"))
    for a, sim in individuales.items():
        filas.append(metrics.summary(sim.equity, sim, f"{a} (estrategia sola)"))
    salidas.csv(pd.DataFrame(filas), f"metricas_{conjunto}.csv")

    curvas = pd.DataFrame({nombres[m]: sim.equity for m, sim in sims.items()})
    curvas["Buy & Hold EW"] = bh
    for a, sim in individuales.items():
        curvas[f"{a} (estrategia sola)"] = sim.equity
    curvas.to_csv(salidas.resultados / f"equity_{conjunto}.csv")

    for nombre, tabla in metrics.returns_table(sims["rp"].equity).items():
        otras = {"EW": metrics.returns_table(sims["ew"].equity)[nombre],
                 "Buy & Hold": metrics.returns_table(bh)[nombre]}
        pd.DataFrame({"RP (sistema)": tabla, **otras}).to_csv(
            salidas.resultados / f"retornos_{nombre}_{conjunto}.csv")

    sims["rp"].operaciones.to_csv(salidas.resultados / f"operaciones_{conjunto}.csv", index=False)
    sims["rp"].rebalanceos.to_csv(salidas.resultados / f"rebalanceos_{conjunto}.csv", index=False)
    pd.DataFrame(sims["rp"].pesos).to_csv(salidas.resultados / f"pesos_{conjunto}.csv")
    pd.DataFrame(senales["s"], index=fechas, columns=config.TICKERS).to_csv(
        salidas.resultados / f"senales_s_{conjunto}.csv")

    # por activo dentro del portafolio (log de operaciones)
    ops = sims["rp"].operaciones
    por_activo = []
    for a in config.TICKERS:
        o = ops[ops["activo"] == a]
        fila = {"activo": a, "pnl_neto": o["pnl_neto"].sum(), "costos": o["costos"].sum()}
        fila.update(metrics.metricas_operaciones(o))
        por_activo.append(fila)
    salidas.csv(pd.DataFrame(por_activo), f"portafolio_por_activo_{conjunto}.csv")

    # por régimen (con bootstrap)
    tablas = [regimes.metricas_por_regimen(sim.equity, regimen, nombres[m]) for m, sim in sims.items()]
    tablas.append(regimes.metricas_por_regimen(bh, regimen, "Buy & Hold EW"))
    salidas.csv(pd.concat(tablas), f"metricas_por_regimen_{conjunto}.csv")
    ops_regimen = []
    for j, etiqueta in enumerate(config.REGIMENES):
        o = ops[ops["regimen_entrada"] == j]
        fila = {"regimen": etiqueta, "pnl_neto": o["pnl_neto"].sum()}
        fila.update(metrics.metricas_operaciones(o))
        ops_regimen.append(fila)
    salidas.csv(pd.DataFrame(ops_regimen), f"operaciones_por_regimen_{conjunto}.csv")
    salidas.json({"conflictos_politica": sims["rp"].conflictos,
                  "exposicion_bruta_max": sims["rp"].exposicion_bruta_max,
                  "costo_anualizado_formula": portfolio.costo_anualizado(
                      float(sims["rp"].rebalanceos["turnover"].mean()) if len(sims["rp"].rebalanceos) else 0.0,
                      len(sims["rp"].rebalanceos) / (len(fechas) / config.DIAS_ANIO)),
                  "costos_cobrados_motor_anual_pct": sims["rp"].costos_totales
                  / (len(fechas) / config.DIAS_ANIO) / config.CAPITAL_INICIAL},
                 f"portafolio_resumen_{conjunto}.json")


def validar_regimenes(salidas, panel, features, resultados, regimen, fechas, sim_rp, conjunto,
                      modelo_final):
    """regimenes_validacion.json y tablas asociadas."""
    persistencia, A = regimes.resumen_persistencia(regimen.loc[fechas])
    salidas.csv(persistencia, f"regimen_persistencia_{conjunto}.csv")
    siluetas = [r["modelo"].silhouette for r in resultados]
    corr_resumen, matrices = regimes.correlacion_por_regimen(panel["close"].loc[fechas],
                                                             regimen.loc[fechas])
    salidas.csv(corr_resumen, "correlacion_por_regimen.csv")
    pd.concat(matrices, names=["regimen"]).to_csv(salidas.resultados / "correlacion_por_regimen_matrices.csv")
    eventos = sim_rp.eventos_regimen
    ops = sim_rp.operaciones
    # todas las features para las figuras, con el régimen
    serie = features.copy()
    serie["regimen"] = regimen
    serie["indice_ew"] = regimes.indice_equiponderado(panel["close"])[0]
    serie.to_csv(salidas.resultados / "regimen_serie.csv")
    # silhouette con las 5 features (para documentar la selección)
    modelo_5 = regimes.ajustar_modelo(features, modelo_final.fin_train, regimes.FEATURES)
    salidas.json({
        "features_usadas": modelo_final.features,
        "silhouette_modelo_congelado": modelo_final.silhouette,
        "silhouette_5_features_misma_fecha": modelo_5.silhouette,
        "silhouette_por_ventana": {"media": float(np.mean(siluetas)), "min": float(np.min(siluetas)),
                                   "max": float(np.max(siluetas))},
        "objetivo_silhouette": 0.4,
        "persistencia": persistencia.to_dict("records"),
        "objetivo_duracion_dias": 10,
        "matriz_transicion": A.tolist(),
        "centroides_modelo_congelado": modelo_final.centroides.to_dict("records"),
        "transiciones": {
            "n_transiciones": int(len(eventos)),
            "posiciones_abiertas_afectadas": int(eventos["posiciones_abiertas"].sum()) if len(eventos) else 0,
            "cierres_por_cambio_a_crisis": int((ops["motivo_salida"] == "cambio de régimen").sum()),
            "reglas": ["La posición abierta conserva su SL/TP de entrada.",
                       "Las entradas nuevas usan el θ del régimen nuevo.",
                       "El m nuevo entra en la siguiente revisión de rebalanceo.",
                       "Si el nuevo régimen es Crisis: al open siguiente se cierran las posiciones "
                       "abiertas con |s| < 1 y el resto se ajusta a m = 0.3 (revisión inmediata)."],
        },
    }, "regimenes_validacion.json")
    eventos.to_csv(salidas.resultados / f"eventos_regimen_{conjunto}.csv", index=False)


def diagnosticos_optuna(salidas, panel, features, ultimo, variante, n_trials):
    """Historia, importancia, slices y superficie 3D del estudio global de un
    activo representativo en la última ventana de TRAIN."""
    import optuna

    v = ultimo["ventana"]
    # activo representativo: el de más trials válidos en el estudio global
    # (se usa el estudio por activo de la misma ventana, que existe para ambas variantes en rolling)
    validos = {a: ultimo["por_activo"][a]["global"]["n_validos"] for a in config.TICKERS}
    activo = max(validos, key=validos.get)
    estudio, evaluar = optimize.estudio_diagnostico(panel, features, v, activo, n_trials)
    trials = estudio.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials["activo"] = activo
    salidas.csv(trials, "optuna_trials_diagnostico.csv")
    validos_trials = [t for t in estudio.trials if t.value is not None
                      and t.value > config.OBJETIVO_INVALIDO]
    importancia = {}
    if len(validos_trials) >= 5:
        solo_validos = optuna.create_study(direction="maximize")
        solo_validos.add_trials(validos_trials)
        importancia = optuna.importance.get_param_importances(solo_validos)
    salidas.csv(pd.DataFrame({"parametro": list(importancia), "importancia": list(importancia.values())}),
                "optuna_importancia.csv")
    theta_rob, _, _, _ = optimize.theta_robusto(estudio)
    p1, p2 = (list(importancia)[:2] + [None, None])[:2]
    salidas.json({"activo": activo, "ventana_k": v["k"], "train_ini": v["train_ini"].date(),
                  "train_fin": v["train_fin"].date(), "parametros_superficie": [p1, p2],
                  "theta_robusto": theta_rob, "n_trials_validos": len(validos_trials)},
                 "optuna_diagnostico.json")
    if theta_rob is None or p2 is None:
        return
    filas = []
    for x in _rejilla(p1):
        for y in _rejilla(p2):
            theta = dict(theta_rob)
            theta[p1], theta[p2] = x, y
            valor, n_ops = evaluar(theta)
            filas.append({p1: x, p2: y, "calmar": valor if n_ops >= config.N_MIN_GLOBAL else np.nan,
                          "n_operaciones": n_ops})
    superficie = pd.DataFrame(filas)
    superficie.attrs = {}
    salidas.csv(superficie, "optuna_superficie.csv")


def _rejilla(parametro, puntos=12):
    tipo, minimo, maximo = optimize.ESPACIO[parametro]
    valores = np.linspace(minimo, maximo, puntos)
    if tipo == "int":
        return sorted(set(int(round(v)) for v in valores))
    return [float(v) for v in valores]


# ----------------------------------------------------------------------------
# Etapa TEST (se toca UNA sola vez, con candado)
# ----------------------------------------------------------------------------
def etapa_test(panel_completo, split, salidas):
    ruta_theta = salidas.resultados / "theta_congelado.json"
    if not ruta_theta.exists():
        raise SystemExit("No existe theta_congelado.json: corre primero --stage train.")
    hash_theta = sha256_archivo(ruta_theta)
    congelado = salidas.leer_json("theta_congelado.json")

    codigo, estado = git("status", "--porcelain")
    if codigo != 0 or estado:
        mensaje = "El working tree de git no está limpio (o no hay repo): " + (estado or "sin repo")
        if not salidas.quick:
            raise SystemExit("ABORTADO. " + mensaje)
        logging.warning("[ensayo --quick] %s (se continúa solo porque es ensayo)", mensaje.split(":")[0])
    codigo, head = git("rev-parse", "HEAD")
    if codigo != 0 and not salidas.quick:
        raise SystemExit("ABORTADO: no hay commit (git rev-parse HEAD falló).")

    ruta_lock = salidas.resultados / "test_lock.json"
    if ruta_lock.exists():
        lock = salidas.leer_json("test_lock.json")
        if lock["sha256_theta_congelado"] != hash_theta:
            raise SystemExit("ABORTADO: theta_congelado.json cambió después de correr el TEST. "
                             "El candado impide re-optimizar después de ver el test.")
    salidas.json({"commit": head if codigo == 0 else "SIN COMMIT (ensayo)",
                  "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "sha256_theta_congelado": hash_theta,
                  "ensayo_quick": salidas.quick}, "test_lock.json")

    panel = truncar(panel_completo, split["n_total"])
    fechas = panel["close"].index
    f_test = fechas[split["n_train"]:split["n_total"]]
    features = regimes.calcular_features(panel)
    modelo = joblib.load(salidas.resultados / congelado["modelo_regimen"]["archivo"])
    if sha256_archivo(salidas.resultados / congelado["modelo_regimen"]["archivo"]) != \
            congelado["modelo_regimen"]["sha256"]:
        raise SystemExit("ABORTADO: el modelo de régimen congelado no coincide con su hash.")
    nombre_a_id = {n: j for j, n in enumerate(config.REGIMENES)}
    thetas = {a: {nombre_a_id[r]: t for r, t in d.items()} for a, d in congelado["thetas"].items()}
    frecuencia = congelado["rebalanceo"]["frecuencia"]
    delta = congelado["rebalanceo"]["delta"]

    # ---- evaluación oficial: sistema congelado ---------------------------
    regimen = regimes.clasificar(features, [(f_test[0], modelo)])
    senales = portfolio.construir_senales(panel, regimen, [(f_test[0], f_test[-1], thetas)], f_test)
    ctx = portfolio.ContextoSimulacion(panel, regimen, f_test)
    sims = {m: ctx.simular(senales, m, frecuencia, delta) for m in ["rp", "naive", "ew"]}
    guardar_conjunto(salidas, "test", panel, sims, senales, regimen, f_test)
    persistencia, _ = regimes.resumen_persistencia(regimen.loc[f_test])
    salidas.csv(persistencia, "regimen_persistencia_test.csv")
    regimen.loc[f_test].to_csv(salidas.resultados / "regimen_serie_test.csv")
    sims["rp"].eventos_regimen.to_csv(salidas.resultados / "eventos_regimen_test.csv", index=False)
    escenarios, slippage = optimize.escenarios_ejecucion(ctx, senales, frecuencia, delta)
    salidas.csv(escenarios, "costos_realistas_test.csv")
    salidas.csv(slippage, "slippage_test.csv")

    # ---- evaluación secundaria: el walk-forward sigue sobre TEST ----------
    n_trials = TRIALS_QUICK if salidas.quick else config.N_TRIALS
    ventanas = [v for v in optimize.generar_ventanas(fechas, split["n_train"], len(fechas) - 1)
                if v["oos_ini"] is not None and v["oos_ini"] >= f_test[0]]
    previas = optimize.correr_walk_forward(
        panel, features, [v for v in optimize.generar_ventanas(fechas, split["n_train"], split["n_train"] - 1)],
        n_trials, [congelado["variante"]], salidas.cache / f"wf_rolling_t{n_trials}") if not salidas.quick else []
    res = optimize.correr_walk_forward(panel, features, ventanas, n_trials, [congelado["variante"]],
                                       salidas.cache / f"wf_test_t{n_trials}")
    # el fallback causal puede heredar el θ de ventanas de TRAIN previas
    optimize.aplicar_fallback_previo(list(previas) + list(res), congelado["variante"])
    regimen_wf = regimes.clasificar(features, optimize.horario_modelos(res))
    f_wf = optimize.fechas_oos(res, fechas)
    horario = optimize.horario_theta(res, congelado["variante"])
    senales_wf = portfolio.construir_senales(panel, regimen_wf, horario, f_wf)
    ctx_wf = portfolio.ContextoSimulacion(panel, regimen_wf, f_wf, ctx.cache_pesos)
    sim_wf = ctx_wf.simular(senales_wf, "rp", frecuencia, delta)
    sim_cong = ctx.simular(senales, "rp", frecuencia, delta)
    filas = [metrics.summary(sim_cong.equity.loc[f_wf[0]:], None, "congelado (oficial)"),
             metrics.summary(sim_wf.equity, sim_wf, "walk-forward re-optimizado (secundario)")]
    salidas.csv(pd.DataFrame(filas), "test_secundario_wf.csv")
    pd.DataFrame({"congelado": sim_cong.equity, "wf_reoptimizado": sim_wf.equity}).to_csv(
        salidas.resultados / "equity_test_secundario.csv")
    salidas.json({"trials_totales_test_secundario": sum(r["n_trials_total"] for r in res),
                  "ventanas": len(res)}, "test_secundario_resumen.json")
    logging.info("Etapa TEST terminada (commit %s)", head if codigo == 0 else "ensayo")


# ----------------------------------------------------------------------------
# Etapa REPORT
# ----------------------------------------------------------------------------
def etapa_report(salidas):
    from src import plots, report
    plots.generar_todas(salidas.resultados, salidas.figuras)
    report.generar_todo(salidas.resultados, salidas.figuras,
                        None if salidas.quick else config.DIR_DOCS, salidas.quick)


def guardar_hashes(salidas):
    """Hashes de los CSV generados (para verificar refactors en --quick)."""
    hashes = {p.name: sha256_archivo(p) for p in sorted(salidas.resultados.glob("*.csv"))}
    with open(salidas.resultados.parent / "hashes.json", "w") as f:
        json.dump(hashes, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Lab 02 MyST — Equipo 2")
    parser.add_argument("--stage", choices=["data", "train", "test", "report", "all"], default="all")
    parser.add_argument("--quick", action="store_true",
                        help="ensayo con 20 trials y pocas ventanas; escribe en .cache/quick/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    salidas = Salidas(args.quick)
    if args.quick:
        ruta_lock = salidas.resultados / "test_lock.json"
        if ruta_lock.exists() and args.stage in ("all", "train"):
            ruta_lock.unlink()            # el ensayo se puede repetir desde cero
    inicio = time.time()
    panel, split = etapa_data(salidas)
    if args.stage in ("train", "all"):
        etapa_train(panel, split, salidas)
    if args.stage in ("test", "all"):
        if args.stage == "all" and not args.quick and not arbol_limpio():
            logging.warning("TEST omitido: el working tree no está limpio. Haz commit y corre "
                            "`python main.py --stage test`.")
        else:
            etapa_test(panel, split, salidas)
    if args.stage in ("report", "all"):
        etapa_report(salidas)
    if args.quick:
        guardar_hashes(salidas)
    logging.info("Listo en %.1f min", (time.time() - inicio) / 60)


if __name__ == "__main__":
    main()
