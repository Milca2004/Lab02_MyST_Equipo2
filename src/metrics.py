"""Métricas de desempeño. Rf = 0 y 252 días hábiles por año."""
import numpy as np
import pandas as pd

from src import config


def rendimientos(equity):
    """R_t = V_t / V_{t−1} − 1."""
    return equity.pct_change().dropna()


def cagr(equity):
    """CAGR = (V_final / V_inicial)^(252/N) − 1, con N = número de días con rendimiento."""
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (config.DIAS_ANIO / n) - 1.0)


def volatilidad_anual(equity):
    """σ_anual = σ(R_t)·√252."""
    r = rendimientos(equity)
    return float(r.std(ddof=1) * np.sqrt(config.DIAS_ANIO)) if len(r) > 1 else 0.0


def sharpe(equity):
    """Sharpe = E[R]/σ·√252  (Rf = 0)."""
    r = rendimientos(equity)
    sigma = r.std(ddof=1) if len(r) > 1 else 0.0
    if sigma == 0 or not np.isfinite(sigma):
        return 0.0
    return float(r.mean() / sigma * np.sqrt(config.DIAS_ANIO))


def sortino(equity):
    """Sortino = E[R]/σ_d·√252,  σ_d = √(1/N·Σ min(0, R_t)²)."""
    r = rendimientos(equity)
    if len(r) == 0:
        return 0.0
    sigma_d = np.sqrt(np.mean(np.minimum(0.0, r) ** 2))
    if sigma_d == 0:
        return 0.0
    return float(r.mean() / sigma_d * np.sqrt(config.DIAS_ANIO))


def serie_drawdown(equity):
    """DD_t = (Peak_t − V_t)/Peak_t, con Peak_t = max_{s≤t} V_s."""
    pico = equity.cummax()
    return (pico - equity) / pico


def max_drawdown(equity):
    """MDD = max_t (Peak_t − V_t)/Peak_t  (positivo)."""
    return float(serie_drawdown(equity).max()) if len(equity) else 0.0


def calmar(equity, piso=config.PISO_MDD):
    """Calmar = CAGR / |MDD|.

    |MDD| tiene un piso de 1%: sin él, una curva casi sin pérdidas daría un
    Calmar que explota (división entre ~0) y el optimizador lo perseguiría.
    """
    return cagr(equity) / max(abs(max_drawdown(equity)), piso)


def duracion_max_drawdown(equity):
    """Máximo número de días consecutivos por debajo del pico previo."""
    bajo_pico = (equity < equity.cummax()).to_numpy()
    maximo, actual = 0, 0
    for abajo in bajo_pico:
        actual = actual + 1 if abajo else 0
        maximo = max(maximo, actual)
    return int(maximo)


def metricas_operaciones(operaciones):
    """Win rate, payoff, profit factor y conteos (sobre PnL neto por operación).

    win rate = #ganadoras / #operaciones
    payoff = promedio ganancia / |promedio pérdida|
    profit factor = Σ ganancias / |Σ pérdidas|
    """
    n = len(operaciones)
    if n == 0:
        return {"n_operaciones": 0, "n_largas": 0, "n_cortas": 0, "win_rate": np.nan,
                "payoff": np.nan, "profit_factor": np.nan}
    pnl = operaciones["pnl_neto"]
    ganadoras = pnl[pnl > 0]
    perdedoras = pnl[pnl <= 0]
    perdida_media = abs(perdedoras.mean()) if len(perdedoras) else np.nan
    suma_perdidas = abs(perdedoras.sum())
    return {
        "n_operaciones": n,
        "n_largas": int((operaciones["direccion"] == "largo").sum()),
        "n_cortas": int((operaciones["direccion"] == "corto").sum()),
        "win_rate": len(ganadoras) / n,
        "payoff": ganadoras.mean() / perdida_media if len(ganadoras) and perdida_media else np.nan,
        "profit_factor": ganadoras.sum() / suma_perdidas if suma_perdidas > 0 else np.nan,
    }


def summary(equity, resultado=None, nombre=""):
    """Una fila con todas las métricas (para las tablas del reporte).

    `resultado` (ResultadoBacktest) agrega operaciones, costos y turnover.
    recovery factor = ganancia neta / (MDD·capital inicial)
    costos/retorno bruto = costos totales / (PnL neto + costos totales)
    """
    mdd = max_drawdown(equity)
    fila = {
        "nombre": nombre,
        "inicio": equity.index[0].date() if len(equity) else None,
        "fin": equity.index[-1].date() if len(equity) else None,
        "retorno_total": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr(equity),
        "volatilidad": volatilidad_anual(equity),
        "sharpe": sharpe(equity),
        "sortino": sortino(equity),
        "mdd": mdd,
        "calmar": calmar(equity),
        "duracion_max_dd": duracion_max_drawdown(equity),
    }
    ganancia = equity.iloc[-1] - equity.iloc[0]
    fila["recovery_factor"] = ganancia / (mdd * equity.iloc[0]) if mdd > 0 else np.nan
    if resultado is not None:
        fila.update(metricas_operaciones(resultado.operaciones))
        costos = resultado.costos_totales
        pnl_bruto = ganancia + costos
        fila["costos_totales"] = costos
        fila["pnl_bruto"] = pnl_bruto
        fila["costos_vs_bruto"] = costos / pnl_bruto if pnl_bruto > 0 else np.nan
        fila["turnover_anual"] = resultado.turnover_anual
        fila["exposicion_media"] = float(resultado.pesos.abs().sum(axis=1).mean())
    return fila


def returns_table(equity):
    """Rendimientos mensuales, trimestrales y anuales (compuestos)."""
    tablas = {}
    for nombre, frecuencia in [("mensual", "ME"), ("trimestral", "QE"), ("anual", "YE")]:
        cierre = equity.resample(frecuencia).last()
        inicio = pd.concat([equity.iloc[:1], cierre.iloc[:-1]])
        inicio.index = cierre.index
        tablas[nombre] = (cierre / inicio.to_numpy() - 1.0).rename("rendimiento")
    return tablas


def tabla_mensual_pivote(equity):
    """Matriz año × mes de rendimientos mensuales (para el heatmap)."""
    mensual = returns_table(equity)["mensual"]
    tabla = pd.DataFrame({"anio": mensual.index.year, "mes": mensual.index.month,
                          "r": mensual.to_numpy()})
    return tabla.pivot(index="anio", columns="mes", values="r")
