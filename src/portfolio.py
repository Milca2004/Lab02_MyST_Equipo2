"""Portafolio multi-activo: Risk Parity, agregación de señales y rebalanceo.

Sigue la estructura de las notas "Fundamentos Matemáticos de Risk Parity:
de R_p al Rebalanceo" (Pasos 1–8). Σ siempre se estima sobre RENDIMIENTOS,
nunca sobre precios (los precios no son estacionarios).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src import config


# ----------------------------------------------------------------------------
# Pasos 1–4: riesgo del portafolio y contribuciones
# ----------------------------------------------------------------------------
def portfolio_vol(w, cov):
    """Pasos 1–2: σ_p = √(wᵀΣw)."""
    w = np.asarray(w, dtype=float)
    return float(np.sqrt(w @ cov @ w))


def marginal_risk_contribution(w, cov):
    """Paso 3 (regla de la cadena): MRC_k = ∂σ_p/∂w_k = (Σw)_k / σ_p."""
    w = np.asarray(w, dtype=float)
    return (cov @ w) / portfolio_vol(w, cov)


def risk_contribution(w, cov):
    """Paso 4: RC_i = w_i·(Σw)_i / σ_p.  Euler: Σ_i RC_i = σ_p (exacto)."""
    w = np.asarray(w, dtype=float)
    return w * marginal_risk_contribution(w, cov)


def risk_contribution_pct(w, cov):
    """RC_i / σ_p: porcentaje genuino del riesgo total (suma 1 por Euler)."""
    return risk_contribution(w, cov) / portfolio_vol(w, cov)


# ----------------------------------------------------------------------------
# Paso 5: tres versiones de pesos
# ----------------------------------------------------------------------------
def pesos_iguales(n):
    """EW: w_i = 1/n."""
    return np.full(n, 1.0 / n)


def pesos_naive(cov):
    """RP naive (volatilidad inversa): w_i = (1/σ_i) / Σ_j (1/σ_j).

    Exacto solo si todas las correlaciones por pares son iguales.
    """
    sigma = np.sqrt(np.diag(cov))
    inversa = 1.0 / sigma
    return inversa / inversa.sum()


def pesos_rp_spinu(cov, tol=config.TOL_RP):
    """RP por optimización con la formulación convexa de Spinu (2013).

        min_{y>0}  ½ yᵀΣy − (1/n)·Σ_i ln(y_i)     y luego   w = y / Σ_j y_j

    En el óptimo, ∇ = Σy − 1/(n·y) = 0  ⇒  y_i(Σy)_i = 1/n para todo i,
    es decir, contribuciones al riesgo idénticas. Es convexa (cuadrática
    convexa + barrera logarítmica convexa): solución única.
    No se minimiza Σ(RC_i − RC_j)² porque ese problema no es convexo.

    Σ se divide entre su diagonal promedio solo para el condicionamiento
    numérico; los pesos finales no cambian con esa escala.
    """
    n = cov.shape[0]
    cov_escalada = cov / np.mean(np.diag(cov))

    def objetivo(y):
        return 0.5 * y @ cov_escalada @ y - np.sum(np.log(y)) / n

    def gradiente(y):
        return cov_escalada @ y - 1.0 / (n * y)

    y0 = pesos_naive(cov_escalada)
    resultado = minimize(objetivo, y0, jac=gradiente, method="L-BFGS-B",
                         bounds=[(1e-10, None)] * n,
                         options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000})
    w = resultado.x / resultado.x.sum()

    # Verificación numérica obligatoria en cada cálculo
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


# ----------------------------------------------------------------------------
# Paso 6: estimadores de Σ
# ----------------------------------------------------------------------------
def cov_muestral(rend):
    """Covarianza muestral S = 1/(T−1)·Σ (r_t − r̄)(r_t − r̄)ᵀ (oficial)."""
    return np.cov(rend, rowvar=False, ddof=1)


def cov_ewma(rend, lam=config.LAMBDA_EWMA):
    """EWMA (RiskMetrics): Σ_t = λΣ_{t−1} + (1 − λ)·r_t r_tᵀ.

    Se inicia con la covarianza muestral de los primeros 20 días y se recorre
    la ventana hacia adelante. T_eff = 1/(1 − λ) ≈ 17 días con λ = 0.94.
    """
    sigma = np.cov(rend[:20], rowvar=False, ddof=1)
    for r in rend[20:]:
        sigma = lam * sigma + (1.0 - lam) * np.outer(r, r)
    return sigma


def cov_ledoit_wolf(rend):
    """Ledoit-Wolf: Σ_shrink = δ*·F + (1 − δ*)·S (sklearn)."""
    return LedoitWolf().fit(rend).covariance_


ESTIMADORES = {"muestral": cov_muestral, "ewma": cov_ewma, "ledoit_wolf": cov_ledoit_wolf}


def serie_pesos(rendimientos, revision, metodo="rp", estimador="muestral",
                ventana=config.VENTANA_COV):
    """Pesos de largo plazo w^RP (o EW/naive) vigentes cada día.

    En cada fecha de revisión t se estima Σ con los rendimientos de
    [t − ventana + 1, t] (solo datos ≤ t) y se recalculan los pesos; entre
    revisiones se mantienen. Al inicio se usa ventana creciente con un mínimo
    de MIN_DIAS_COV días.

    Regresa (pesos DataFrame T×n, correlaciones array T×n×n, numero_condicion Series).
    """
    datos = rendimientos.to_numpy()
    T, n = datos.shape
    pesos = np.full((T, n), np.nan)
    corr = np.full((T, n, n), np.nan)
    condicion = np.full(T, np.nan)
    w_vigente, corr_vigente, cond_vigente = None, None, np.nan
    for t in range(T):
        historia = t  # rendimientos válidos hasta t (el del día 0 es NaN)
        if revision[t] or (w_vigente is None and historia >= config.MIN_DIAS_COV):
            if historia >= config.MIN_DIAS_COV:
                inicio = max(1, t + 1 - ventana)
                ventana_rend = datos[inicio:t + 1]
                cov = ESTIMADORES[estimador](ventana_rend)
                w_vigente = calcular_pesos(cov, metodo)
                sigma = np.sqrt(np.diag(cov))
                corr_vigente = cov / np.outer(sigma, sigma)
                cond_vigente = np.linalg.cond(cov)
        if w_vigente is not None:
            pesos[t] = w_vigente
            corr[t] = corr_vigente
            condicion[t] = cond_vigente
    indice = rendimientos.index
    return (pd.DataFrame(pesos, index=indice, columns=rendimientos.columns),
            corr, pd.Series(condicion, index=indice))


# ----------------------------------------------------------------------------
# Paso 7: agregación de señales
# ----------------------------------------------------------------------------
def resolver_conflictos(s, corr, umbral=config.UMBRAL_CORR_CONFLICTO):
    """Política de conflictos entre activos correlacionados.

    Si corr(i, j) > umbral y s_i, s_j tienen signo opuesto: se queda el de
    mayor |s| y el otro va a 0; si empatan, ambos se multiplican por 0.5.
    Las decisiones se toman con los s originales (no depende del orden).
    Regresa (s_ajustada, numero_de_conflictos).
    """
    s = np.asarray(s, dtype=float)
    if corr is None:
        return s.copy(), 0
    n = len(s)
    anular, mitad = set(), set()
    conflictos = 0
    for i in range(n):
        for j in range(i + 1, n):
            opuestos = s[i] * s[j] < 0
            if opuestos and corr[i, j] > umbral:
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
    """Paso 7: composición de pesos de largo plazo con señales.

        w̃_i = w_i^RP · s_i
        w^target = m(régimen) · w̃ / max(1, Σ_i |w̃_i|)

    max(1, Σ|w̃|) garantiza que nunca haya apalancamiento.
    Regresa (w_target, numero_de_conflictos).
    """
    s_ajustada, conflictos = resolver_conflictos(s, corr, umbral)
    w_tilde = np.asarray(w_rp, dtype=float) * s_ajustada
    bruto = np.sum(np.abs(w_tilde))
    w_target = m * w_tilde / max(1.0, bruto)
    return w_target, conflictos


# ----------------------------------------------------------------------------
# Paso 8: rebalanceo y turnover
# ----------------------------------------------------------------------------
def pesos_despues_drift(w, rend):
    """Peso justo antes de rebalancear (después del movimiento de precios).

    w_{i,t⁻} = w_i(1 + r_i) / Σ_j w_j(1 + r_j)
    (se supone que todo el capital está invertido en los activos)
    """
    w = np.asarray(w, dtype=float)
    valor = w * (1.0 + np.asarray(rend, dtype=float))
    return valor / valor.sum()


def turnover(w_nuevo, w_antes):
    """T_t = ½·Σ_i |w_{i,t} − w_{i,t⁻}|, con w_{t⁻} el peso post-drift."""
    return 0.5 * float(np.sum(np.abs(np.asarray(w_nuevo) - np.asarray(w_antes))))


def costo_anualizado(turnover_promedio, rebalanceos_por_anio, comision=config.COMISION):
    """cost_ann ≈ T̄ × f × 2c."""
    return turnover_promedio * rebalanceos_por_anio * 2.0 * comision


def fechas_revision(n_dias, cada):
    """Fechas de revisión del calendario: cada `cada` días sobre el índice global."""
    revision = np.zeros(n_dias, dtype=bool)
    revision[::cada] = True
    return revision


# ----------------------------------------------------------------------------
# Simulación del portafolio (une señales, régimen, pesos y el motor)
# ----------------------------------------------------------------------------
def construir_senales(panel, regimen, horario_theta, fechas_sim, umbral=None, indicadores=None):
    """Arreglos (T × n) de señal y parámetros vigentes para la simulación.

    horario_theta: lista de (desde, hasta, thetas) con thetas[activo][régimen] = θ.
    En el día t se usa el θ del régimen confirmado en t (θ del régimen nuevo
    para entradas nuevas). Las señales se calculan con datos ≤ `hasta`
    (causales: el valor en t solo usa datos ≤ t).
    """
    from src.data import asset_frame
    from src.signals import generate_signals

    if umbral is None:
        umbral = config.UMBRAL_CONFIRMACION
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
                senal = generate_signals(ohlcv, theta, umbral, indicadores)
                senal = senal["s"].reindex(fechas_sim).fillna(0.0).to_numpy()
                arreglos["s"][dias, i] = senal[dias]
                arreglos["m_sl"][dias, i] = theta["m_sl"]
                arreglos["m_tp"][dias, i] = theta["m_tp"]
                arreglos["max_hold"][dias, i] = theta["max_hold"]
    atr = pd.DataFrame({a: _atr_activo(panel, a) for a in config.TICKERS})
    arreglos["atr"] = atr.reindex(fechas_sim).to_numpy()
    return arreglos


def _atr_activo(panel, activo):
    from src.signals import atr
    return atr(panel["high"][activo], panel["low"][activo], panel["close"][activo])


def simular_portafolio(panel, senales, regimen, pesos_largo_plazo, fechas_sim,
                       frecuencia=config.FRECUENCIA_DEFAULT, delta=config.DELTA_DEFAULT,
                       costos=None, m_regimen=None, usar_regimen=True, cerrar_al_final=False):
    """Corre el motor para el portafolio de 6 activos en `fechas_sim`.

    pesos_largo_plazo: salida de serie_pesos (pesos, correlaciones, condición)
    calculada sobre TODO el calendario (cada fila usa solo datos ≤ t).
    """
    from src.backtest import DatosMotor, run_backtest

    todas = panel["close"].index
    posicion = todas.get_indexer(fechas_sim)
    pesos, corr = pesos_largo_plazo[0], pesos_largo_plazo[1]
    cada = config.FRECUENCIAS_REBALANCEO[frecuencia]
    revision = (posicion % cada) == 0                  # rejilla del calendario global
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
        revision=revision, delta=delta, m_regimen=m_regimen,
        cerrar_al_final=cerrar_al_final)
    return run_backtest(datos, costos)


def simular_activo_individual(panel, senales, i, fechas_sim, costos=None):
    """Estrategia de un solo activo con el 100% del capital (n = 1, peso 1)."""
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
    """Buy & Hold equiponderado: compra al open del primer día (paga comisión),
    mantiene y vende al cierre del último día (paga comisión)."""
    abre = panel["open"].loc[fechas_sim[0]].to_numpy()
    cierres = panel["close"].loc[fechas_sim].to_numpy()
    n = len(abre)
    cantidades = (capital / n) / (abre * (1.0 + comision))
    valor = cierres @ cantidades
    valor[-1] = valor[-1] * (1.0 - comision)
    return pd.Series(valor, index=fechas_sim, name="buy_and_hold")


class ContextoSimulacion:
    """Agrupa lo que se repite en todas las simulaciones de un tramo: panel,
    serie de régimen, fechas y un caché de pesos de largo plazo (los pesos
    solo dependen de método, estimador, ventana y frecuencia de revisión)."""

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


# ----------------------------------------------------------------------------
# Análisis del Paso 8 y comparaciones
# ----------------------------------------------------------------------------
def descomponer_resultado(resultado, capital=config.CAPITAL_INICIAL):
    """Retorno bruto, costo total y retorno neto (como fracción del capital)."""
    neto = resultado.equity.iloc[-1] / capital - 1.0
    costo = resultado.costos_totales / capital
    return {"retorno_bruto": neto + costo, "costo_total": costo, "retorno_neto": neto,
            "turnover_anual": resultado.turnover_anual,
            "n_rebalanceos": len(resultado.rebalanceos)}


def barrido_rebalanceo(contexto, senales):
    """Barrido f × δ (solo en TRAIN, WF-OOS). Debe existir un óptimo interior
    entre no rebalancear (RP deja de serlo) y rebalancear siempre (costo)."""
    from src.metrics import calmar, sharpe, max_drawdown

    filas = []
    for frecuencia in config.FRECUENCIAS_REBALANCEO:
        for delta in config.DELTAS_REBALANCEO:
            res = contexto.simular(senales, "rp", frecuencia, delta)
            fila = {"frecuencia": frecuencia, "delta": delta}
            fila.update(descomponer_resultado(res))
            fila.update({"calmar": calmar(res.equity), "sharpe": sharpe(res.equity),
                         "mdd": max_drawdown(res.equity)})
            turnovers = res.rebalanceos["turnover"]
            fila["turnover_medio_rebalanceo"] = float(turnovers.mean()) if len(turnovers) else 0.0
            filas.append(fila)
    return pd.DataFrame(filas)


def contribuciones_riesgo(rendimientos, fechas, cada=config.FRECUENCIAS_REBALANCEO["mensual"],
                          ventana=config.VENTANA_COV):
    """RC_i/σ_p por activo bajo EW, naive y RP en fechas de revisión (Σ muestral ≤ t).

    Regresa un DataFrame largo: fecha, metodo, activo, peso, rc_pct.
    """
    datos = rendimientos.to_numpy()
    posiciones = rendimientos.index.get_indexer(fechas)
    filas = []
    for t in posiciones[::cada]:
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
    """Pesos RP bajo Σ muestral, EWMA y Ledoit-Wolf: estabilidad y turnover implícito.

    turnover implícito por revisión = ½Σ|w_t − w_{t−1}| entre revisiones consecutivas;
    anual = promedio × revisiones por año; costo ≈ T̄ × f × 2c.
    """
    cada = config.FRECUENCIAS_REBALANCEO[frecuencia]
    revision = fechas_revision(len(rendimientos), cada)
    series, filas = {}, []
    for estimador in ESTIMADORES:
        pesos, _, condicion = serie_pesos(rendimientos, revision, "rp", estimador)
        pesos = pesos.loc[fechas]
        en_revision = pesos[revision[rendimientos.index.get_indexer(fechas)]]
        cambios = 0.5 * en_revision.diff().abs().sum(axis=1).iloc[1:]
        revisiones_anio = config.DIAS_ANIO / cada
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
