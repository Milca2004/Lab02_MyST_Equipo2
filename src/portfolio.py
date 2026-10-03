"""Portafolio multi-activo: Risk Parity, agregación de señales y rebalanceo.

Sigue las notas "Fundamentos Matemáticos de Risk Parity" (pasos 1 a 8). Σ siempre se
estima sobre rendimientos, nunca sobre precios.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src import config


# --- Pasos 1-4: riesgo del portafolio y contribuciones ------------------------
def portfolio_vol(w, cov):
    """σ_p = √(wᵀΣw)."""
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(w @ cov @ w))


def marginal_risk_contribution(w, cov):
    """Contribución marginal: MRC_k = (Σw)_k / σ_p."""
    w = np.asarray(w, dtype=float)
    return (cov @ w) / portfolio_vol(w, cov)


def risk_contribution(w, cov):
    """RC_i = w_i·(Σw)_i / σ_p. Por Euler, la suma de todas es σ_p."""
    w = np.asarray(w, dtype=float)
    return w * marginal_risk_contribution(w, cov)


def risk_contribution_pct(w, cov):
    """RC_i / σ_p: porcentaje del riesgo total que aporta cada activo (suma 1)."""
    return risk_contribution(w, cov) / portfolio_vol(w, cov)


# --- Paso 5: tres versiones de pesos ------------------------------------------
def pesos_iguales(n):
    """EW: w_i = 1/n."""
    return np.full(n, 1.0 / n)


def pesos_naive(cov):
    """RP naive (volatilidad inversa). Solo es exacto si todas las correlaciones son iguales."""
    inversa = 1.0 / np.sqrt(np.diag(cov))
    return inversa / inversa.sum()


def pesos_rp_spinu(cov, tol=config.TOL_RP):
    """Risk Parity con la formulación convexa de Spinu (2013).

        min_{y>0}  ½ yᵀΣy − (1/n)·Σ ln(y_i),   w = y / Σ y

    En el óptimo y_i(Σy)_i = 1/n para todo i, o sea contribuciones iguales. Es convexa, así
    que la solución es única (minimizar Σ(RC_i − RC_j)² no lo sería). Σ se escala por su
    diagonal promedio solo por condicionamiento numérico; no cambia los pesos.
    """
    n = cov.shape[0]
    cov_escalada = cov / np.mean(np.diag(cov))

    def objetivo(y):
        return 0.5 * y @ cov_escalada @ y - np.sum(np.log(y)) / n

    def gradiente(y):
        return cov_escalada @ y - 1.0 / (n * y)

    resultado = minimize(objetivo, pesos_naive(cov_escalada), jac=gradiente, method="L-BFGS-B",
                         bounds=[(1e-10, None)] * n,
                         options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000})
    w = resultado.x / resultado.x.sum()

    error = np.max(np.abs(risk_contribution_pct(w, cov) - 1.0 / n))
    if error >= tol:
        raise ValueError(f"Risk Parity no convergió: max|RC_i/σ_p − 1/n| = {error:.2e}")
    return w


def calcular_pesos(cov, metodo):
    """Pesos según el método: 'ew', 'naive' o 'rp'."""
    if metodo == "ew":
        return pesos_iguales(cov.shape[0])
    if metodo == "naive":
        return pesos_naive(cov)
    if metodo == "rp":
        return pesos_rp_spinu(cov)
    raise ValueError(f"Método de pesos desconocido: {metodo}")


# --- Paso 6: estimadores de Σ -------------------------------------------------
def cov_muestral(rend):
    """Covarianza muestral (el estimador oficial)."""
    return np.cov(rend, rowvar=False, ddof=1)


def cov_ewma(rend, lam=config.LAMBDA_EWMA):
    """EWMA (RiskMetrics): Σ_t = λΣ_{t−1} + (1 − λ)·r_t r_tᵀ.

    Arranca con la covarianza muestral de los primeros 20 días. Con λ = 0.94 equivale a
    unos 17 días efectivos.
    """
    sigma = np.cov(rend[:20], rowvar=False, ddof=1)
    for r in rend[20:]:
        sigma = lam * sigma + (1.0 - lam) * np.outer(r, r)
    return sigma


def cov_ledoit_wolf(rend):
    """Ledoit-Wolf: mezcla de la muestral con un objetivo estructurado (sklearn)."""
    return LedoitWolf().fit(rend).covariance_


ESTIMADORES = {"muestral": cov_muestral, "ewma": cov_ewma, "ledoit_wolf": cov_ledoit_wolf}


def serie_pesos(rendimientos, revision, metodo="rp", estimador="muestral",
                ventana=config.VENTANA_COV):
    """Pesos de largo plazo (RP, EW o naive) vigentes cada día.

    En cada fecha de revisión se estima Σ con la ventana que termina ahí (solo datos hasta
    t) y se recalculan los pesos; entre revisiones se mantienen. Al inicio la ventana crece
    hasta tener MIN_DIAS_COV días.
    Devuelve (pesos T×n, correlaciones T×n×n, número de condición de Σ).
    """
    datos = rendimientos.to_numpy()
    T, n = datos.shape
    pesos = np.full((T, n), np.nan)
    corr = np.full((T, n, n), np.nan)
    condicion = np.full(T, np.nan)
    vigente = None  # (pesos, correlación, condición) de la última revisión
    for t in range(T):
        # el rendimiento del día 0 es NaN, así que hay t rendimientos válidos hasta t
        if t >= config.MIN_DIAS_COV and (revision[t] or vigente is None):
            cov = ESTIMADORES[estimador](datos[max(1, t + 1 - ventana):t + 1])
            sigma = np.sqrt(np.diag(cov))
            vigente = (calcular_pesos(cov, metodo), cov / np.outer(sigma, sigma),
                       np.linalg.cond(cov))
        if vigente is not None:
            pesos[t], corr[t], condicion[t] = vigente
    indice = rendimientos.index
    return (pd.DataFrame(pesos, index=indice, columns=rendimientos.columns),
            corr, pd.Series(condicion, index=indice))


# --- Paso 7: agregación de señales --------------------------------------------
def resolver_conflictos(s, corr, umbral=config.UMBRAL_CORR_CONFLICTO):
    """Política para señales opuestas en activos muy correlacionados.

    Si corr(i, j) > umbral y s_i, s_j tienen signo contrario, se queda la de mayor |s| y la
    otra pasa a 0; si empatan, ambas se reducen a la mitad. Todo se decide con las señales
    originales, así que no depende del orden. Devuelve (señales ajustadas, # de conflictos).
    """
    s = np.asarray(s, dtype=float)
    if corr is None:
        return s.copy(), 0
    anular, mitad = set(), set()
    conflictos = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if s[i] * s[j] < 0 and corr[i, j] > umbral:
                conflictos += 1
                if abs(s[i]) > abs(s[j]):
                    anular.add(j)
                elif abs(s[i]) < abs(s[j]):
                    anular.add(i)
                else:
                    mitad.update([i, j])
    ajustada = s.copy()
    for i in mitad - anular:
        ajustada[i] *= 0.5
    for i in anular:
        ajustada[i] = 0.0
    return ajustada, conflictos


def pesos_objetivo(w_rp, s, m, corr=None, umbral=config.UMBRAL_CORR_CONFLICTO):
    """Combina los pesos de largo plazo con las señales.

        w̃ = w^RP · s        w^target = m(régimen) · w̃ / max(1, Σ|w̃|)

    El max(1, ·) evita el apalancamiento. Devuelve (w_target, # de conflictos).
    """
    s_ajustada, conflictos = resolver_conflictos(s, corr, umbral)
    w_tilde = np.asarray(w_rp, dtype=float) * s_ajustada
    return m * w_tilde / max(1.0, np.sum(np.abs(w_tilde))), conflictos


# --- Paso 8: rebalanceo y turnover --------------------------------------------
def pesos_despues_drift(w, rend):
    """Pesos justo antes de rebalancear, tras el movimiento de precios del día."""
    valor = np.asarray(w, dtype=float) * (1.0 + np.asarray(rend, dtype=float))
    return valor / valor.sum()


def turnover(w_nuevo, w_antes):
    """T = ½·Σ|w_nuevo − w_antes|, con w_antes el peso después del drift."""
    return 0.5 * float(np.sum(np.abs(np.asarray(w_nuevo) - np.asarray(w_antes))))


def costo_anualizado(turnover_promedio, rebalanceos_por_anio, comision=config.COMISION):
    """Costo anual aproximado: T̄ × f × 2c."""
    return turnover_promedio * rebalanceos_por_anio * 2.0 * comision


def fechas_revision(n_dias, cada):
    """Máscara de revisión: True cada `cada` días sobre el calendario global."""
    revision = np.zeros(n_dias, dtype=bool)
    revision[::cada] = True
    return revision


# --- Simulación del portafolio (señales, régimen, pesos y motor) --------------
def construir_senales(panel, regimen, horario_theta, fechas_sim, umbral=None, indicadores=None):
    """Arreglos (T × n) de señal y parámetros vigentes para la simulación.

    horario_theta: lista de (desde, hasta, thetas) con thetas[activo][régimen] = θ. Cada día
    se usa el θ del régimen confirmado ese día. Las señales son causales: usan datos hasta
    `hasta`, y el valor en t solo depende de datos hasta t.
    """
    from src.data import asset_frame
    from src.signals import atr, generate_signals

    umbral = config.UMBRAL_CONFIRMACION if umbral is None else umbral
    T, n = len(fechas_sim), len(config.TICKERS)
    arreglos = {nombre: np.zeros((T, n)) for nombre in ["s", "m_sl", "m_tp", "max_hold"]}
    arreglos["max_hold"][:] = 1
    regimen_sim = regimen.reindex(fechas_sim).to_numpy()
    for desde, hasta, thetas in horario_theta:
        en_tramo = (fechas_sim >= desde) & (fechas_sim <= hasta)
        if not en_tramo.any():
            continue
        for i, activo in enumerate(config.TICKERS):
            ohlcv = asset_frame(panel, activo).loc[:hasta]
            for j in range(config.N_REGIMENES):
                dias = en_tramo & (regimen_sim == j)
                if not dias.any():
                    continue
                theta = thetas[activo][j]
                senal = generate_signals(ohlcv, theta, umbral, indicadores)["s"]
                arreglos["s"][dias, i] = senal.reindex(fechas_sim).fillna(0.0).to_numpy()[dias]
                for clave in ["m_sl", "m_tp", "max_hold"]:
                    arreglos[clave][dias, i] = theta[clave]
    arreglos["atr"] = pd.DataFrame(
        {a: atr(panel["high"][a], panel["low"][a], panel["close"][a])
         for a in config.TICKERS}).reindex(fechas_sim).to_numpy()
    return arreglos


def simular_portafolio(panel, senales, regimen, pesos_largo_plazo, fechas_sim,
                       frecuencia=config.FRECUENCIA_DEFAULT, delta=config.DELTA_DEFAULT,
                       costos=None, m_regimen=None, usar_regimen=True, cerrar_al_final=False):
    """Corre el motor con los activos del portafolio sobre `fechas_sim`.

    pesos_largo_plazo es la salida de serie_pesos, calculada sobre todo el calendario
    (cada fila solo usa datos hasta t).
    """
    from src.backtest import DatosMotor, run_backtest

    posicion = panel["close"].index.get_indexer(fechas_sim)
    pesos, corr = pesos_largo_plazo[0], pesos_largo_plazo[1]
    cada = config.FRECUENCIAS_REBALANCEO[frecuencia]
    datos = DatosMotor(
        fechas=fechas_sim, activos=list(config.TICKERS),
        open=panel["open"].loc[fechas_sim].to_numpy(),
        high=panel["high"].loc[fechas_sim].to_numpy(),
        low=panel["low"].loc[fechas_sim].to_numpy(),
        close=panel["close"].loc[fechas_sim].to_numpy(),
        volumen=panel["volume"].loc[fechas_sim].to_numpy(),
        s=senales["s"], atr=senales["atr"], m_sl=senales["m_sl"], m_tp=senales["m_tp"],
        max_hold=senales["max_hold"],
        regimen=regimen.reindex(fechas_sim).to_numpy() if usar_regimen else None,
        w_rp=pesos.loc[fechas_sim].to_numpy(), corr=corr[posicion],
        revision=(posicion % cada) == 0,  # rejilla del calendario global
        delta=delta, m_regimen=m_regimen, cerrar_al_final=cerrar_al_final)
    return run_backtest(datos, costos)


def simular_activo_individual(panel, senales, i, fechas_sim, costos=None):
    """Estrategia de un solo activo con el 100% del capital."""
    from src.backtest import DatosMotor, run_backtest

    activo = config.TICKERS[i]

    def col(campo):
        return panel[campo][activo].loc[fechas_sim].to_numpy().reshape(-1, 1)

    datos = DatosMotor(
        fechas=fechas_sim, activos=[activo], open=col("open"), high=col("high"),
        low=col("low"), close=col("close"), volumen=col("volume"),
        s=senales["s"][:, [i]], atr=senales["atr"][:, [i]], m_sl=senales["m_sl"][:, [i]],
        m_tp=senales["m_tp"][:, [i]], max_hold=senales["max_hold"][:, [i]])
    return run_backtest(datos, costos)


def buy_and_hold(panel, fechas_sim, comision=config.COMISION, capital=config.CAPITAL_INICIAL):
    """Buy & Hold equiponderado: compra al open del primer día, vende al cierre del último,
    pagando comisión en ambos extremos."""
    abre = panel["open"].loc[fechas_sim[0]].to_numpy()
    cierres = panel["close"].loc[fechas_sim].to_numpy()
    cantidades = (capital / len(abre)) / (abre * (1.0 + comision))
    valor = cierres @ cantidades
    valor[-1] *= 1.0 - comision
    return pd.Series(valor, index=fechas_sim, name="buy_and_hold")


class ContextoSimulacion:
    """Lo que se repite en todas las simulaciones de un tramo: panel, régimen, fechas y un
    caché de pesos (dependen solo de método, estimador, ventana y frecuencia de revisión)."""

    def __init__(self, panel, regimen, fechas, cache_pesos=None):
        self.panel = panel
        self.regimen = regimen
        self.fechas = fechas
        self.cache_pesos = {} if cache_pesos is None else cache_pesos

    def pesos(self, metodo="rp", frecuencia=config.FRECUENCIA_DEFAULT,
              estimador="muestral", ventana=config.VENTANA_COV):
        clave = (metodo, frecuencia, estimador, ventana)
        if clave not in self.cache_pesos:
            rend = self.panel["close"].pct_change()
            revision = fechas_revision(len(rend), config.FRECUENCIAS_REBALANCEO[frecuencia])
            self.cache_pesos[clave] = serie_pesos(rend, revision, metodo, estimador, ventana)
        return self.cache_pesos[clave]

    def simular(self, senales, metodo="rp", frecuencia=config.FRECUENCIA_DEFAULT,
                delta=config.DELTA_DEFAULT, costos=None, m_regimen=None,
                ventana=config.VENTANA_COV, cerrar_al_final=False):
        pesos = self.pesos(metodo, frecuencia, "muestral", ventana)
        return simular_portafolio(self.panel, senales, self.regimen, pesos, self.fechas,
                                  frecuencia, delta, costos, m_regimen,
                                  cerrar_al_final=cerrar_al_final)


# --- Análisis de rebalanceo y comparaciones -----------------------------------
def descomponer_resultado(resultado, capital=config.CAPITAL_INICIAL):
    """Retorno bruto, costo total y retorno neto, como fracción del capital."""
    neto = resultado.equity.iloc[-1] / capital - 1.0
    costo = resultado.costos_totales / capital
    return {"retorno_bruto": neto + costo, "costo_total": costo, "retorno_neto": neto,
            "turnover_anual": resultado.turnover_anual,
            "n_rebalanceos": len(resultado.rebalanceos)}


def barrido_rebalanceo(contexto, senales):
    """Barrido de frecuencia × banda δ (solo en TRAIN, walk-forward OOS).

    Se espera un óptimo interior: no rebalancear deja de ser RP y rebalancear siempre
    cuesta demasiado.
    """
    from src.metrics import calmar, max_drawdown, sharpe

    filas = []
    for frecuencia in config.FRECUENCIAS_REBALANCEO:
        for delta in config.DELTAS_REBALANCEO:
            res = contexto.simular(senales, "rp", frecuencia, delta)
            turnovers = res.rebalanceos["turnover"]
            filas.append({
                "frecuencia": frecuencia, "delta": delta, **descomponer_resultado(res),
                "calmar": calmar(res.equity), "sharpe": sharpe(res.equity),
                "mdd": max_drawdown(res.equity),
                "turnover_medio_rebalanceo": float(turnovers.mean()) if len(turnovers) else 0.0})
    return pd.DataFrame(filas)


def contribuciones_riesgo(rendimientos, fechas, cada=config.FRECUENCIAS_REBALANCEO["mensual"],
                          ventana=config.VENTANA_COV):
    """RC_i/σ_p por activo bajo EW, naive y RP, en fechas de revisión (Σ muestral hasta t).

    Devuelve un DataFrame largo: fecha, metodo, activo, peso, rc_pct.
    """
    datos = rendimientos.to_numpy()
    filas = []
    for t in rendimientos.index.get_indexer(fechas)[::cada]:
        if t < config.MIN_DIAS_COV:
            continue
        cov = cov_muestral(datos[max(1, t + 1 - ventana):t + 1])
        for metodo in ["ew", "naive", "rp"]:
            w = calcular_pesos(cov, metodo)
            rc = risk_contribution_pct(w, cov)
            for i, activo in enumerate(rendimientos.columns):
                filas.append({"fecha": rendimientos.index[t], "metodo": metodo, "activo": activo,
                              "peso": w[i], "rc_pct": rc[i]})
    return pd.DataFrame(filas)


def comparar_estimadores(rendimientos, fechas, frecuencia=config.FRECUENCIA_DEFAULT):
    """Pesos RP con Σ muestral, EWMA y Ledoit-Wolf: estabilidad y turnover implícito.

    El turnover por revisión es ½Σ|w_t − w_{t−1}| entre revisiones consecutivas; el anual
    es ese promedio por las revisiones al año, y el costo es T̄ × f × 2c.
    """
    cada = config.FRECUENCIAS_REBALANCEO[frecuencia]
    revision = fechas_revision(len(rendimientos), cada)
    revisiones_anio = config.DIAS_ANIO / cada
    series, filas = {}, []
    for estimador in ESTIMADORES:
        pesos, _, condicion = serie_pesos(rendimientos, revision, "rp", estimador)
        pesos = pesos.loc[fechas]
        en_revision = pesos[revision[rendimientos.index.get_indexer(fechas)]]
        cambios = 0.5 * en_revision.diff().abs().sum(axis=1).iloc[1:]
        filas.append({
            "estimador": estimador,
            "volatilidad_media_pesos": float(pesos.std().mean()),
            "turnover_medio_revision": float(cambios.mean()),
            "turnover_anual_implicito": float(cambios.mean() * revisiones_anio),
            "costo_anual_implicito": costo_anualizado(cambios.mean(), revisiones_anio),
            "numero_condicion_medio": float(condicion.loc[fechas].mean()),
        })
        series[estimador] = pesos
    return pd.DataFrame(filas), series
