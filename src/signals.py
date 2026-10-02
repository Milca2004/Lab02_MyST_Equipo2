"""Indicadores técnicos (implementados a mano) y regla de confirmación 2 de 3.

Causalidad: todo indicador en el día t usa solo datos hasta el cierre de t.
Solo se usan medias recursivas (ewm con adjust=False) y ventanas móviles hacia
atrás (rolling). Prohibido shift(-k), center=True o normalizar con toda la serie.
La ejecución de cualquier señal ocurre al open de t+1 (lo hace backtest.py).
"""
import numpy as np
import pandas as pd

from src import config

NOMBRES_VOTOS = ["voto_ema", "voto_rsi", "voto_bb"]


def ema(precio, h):
    """Media móvil exponencial.

    EMA_t = α·P_t + (1 − α)·EMA_{t−1},  con  α = 2/(h + 1).
    """
    alpha = 2.0 / (h + 1.0)
    return precio.ewm(alpha=alpha, adjust=False).mean()


def wilder(serie, n):
    """Suavizado de Wilder: X_t = X_{t−1} + (x_t − X_{t−1})/n  (α = 1/n)."""
    return serie.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close, n):
    """RSI de Wilder.

    ΔP_t = P_t − P_{t−1};  U = max(ΔP, 0);  D = max(−ΔP, 0)
    RS = Wilder(U, n) / Wilder(D, n);  RSI = 100 − 100/(1 + RS)
    """
    delta = close.diff()
    subida = delta.clip(lower=0)
    bajada = (-delta).clip(lower=0)
    media_subida = wilder(subida, n)
    media_bajada = wilder(bajada, n)
    rs = media_subida / media_bajada
    valor = 100.0 - 100.0 / (1.0 + rs)
    valor[media_bajada == 0] = 100.0          # solo subidas -> RSI = 100
    valor.iloc[:n] = np.nan                   # periodo de calentamiento
    return valor


def bollinger(close, n, k):
    """Bandas de Bollinger.

    MB = SMA_N(P);  UB = MB + k·σ_N;  LB = MB − k·σ_N
    (σ_N = desviación estándar móvil de N días, con ddof=0)
    """
    banda_media = close.rolling(n).mean()
    sigma = close.rolling(n).std(ddof=0)
    banda_sup = banda_media + k * sigma
    banda_inf = banda_media - k * sigma
    return banda_media, banda_sup, banda_inf


def atr(high, low, close, n=config.VENTANA_ATR):
    """Average True Range de Wilder.

    TR_t = max(H_t − L_t, |H_t − C_{t−1}|, |L_t − C_{t−1}|);  ATR = Wilder(TR, n)
    """
    cierre_previo = close.shift(1)
    rango1 = high - low
    rango2 = (high - cierre_previo).abs()
    rango3 = (low - cierre_previo).abs()
    tr = pd.concat([rango1, rango2, rango3], axis=1).max(axis=1)
    valor = wilder(tr, n)
    valor.iloc[:n] = np.nan
    return valor


# ----------------------------------------------------------------------------
# Votos individuales x_{i,j} ∈ {−1, 0, +1}
# ----------------------------------------------------------------------------
def voto_ema(close, rapida, lenta):
    """Tendencia: +1 si EMA_rápida > EMA_lenta, −1 si es menor."""
    diferencia = ema(close, rapida) - ema(close, lenta)
    voto = np.sign(diferencia)
    voto.iloc[:lenta] = 0.0                   # calentamiento de la EMA lenta
    return voto


def voto_rsi(close, n, umbral_inf, umbral_sup):
    """Momento: +1 si RSI < umbral_inf (sobreventa), −1 si RSI > umbral_sup."""
    valor = rsi(close, n)
    voto = pd.Series(0.0, index=close.index)
    voto[valor < umbral_inf] = 1.0
    voto[valor > umbral_sup] = -1.0
    return voto


def voto_bollinger(close, n, k):
    """Volatilidad: +1 si close < LB, −1 si close > UB, 0 dentro de las bandas."""
    _, banda_sup, banda_inf = bollinger(close, n, k)
    voto = pd.Series(0.0, index=close.index)
    voto[close < banda_inf] = 1.0
    voto[close > banda_sup] = -1.0
    return voto


def calcular_votos(ohlcv, params):
    """DataFrame con los 3 votos individuales (columnas NOMBRES_VOTOS)."""
    close = ohlcv["close"]
    lenta = params["ema_fast"] + params["ema_gap"]
    votos = pd.DataFrame({
        "voto_ema": voto_ema(close, params["ema_fast"], lenta),
        "voto_rsi": voto_rsi(close, params["rsi_window"], params["rsi_lower"],
                             params["rsi_upper"]),
        "voto_bb": voto_bollinger(close, params["bb_window"], params["bb_k"]),
    })
    return votos


def confirmar(votos, umbral=config.UMBRAL_CONFIRMACION):
    """Regla de confirmación (notas de Risk Parity, Paso 7.2).

    Con k indicadores que votan x_{i,j} ∈ {−1, 0, +1}:
        s_i = (1/k)·Σ_j x_{i,j}   si |Σ_j x_{i,j}| ≥ umbral
        s_i = 0                   en otro caso
    Con k = 3 y umbral = 2, s_i ∈ {−1, −2/3, 0, +2/3, +1}.
    Caso borde: (+1, +1, −1) da Σ = 1 < 2 -> s_i = 0 (no se abre).
    """
    k = votos.shape[1]
    suma = votos.sum(axis=1)
    s = suma / k
    s[suma.abs() < umbral] = 0.0
    return s


def generate_signals(ohlcv, params, umbral=config.UMBRAL_CONFIRMACION,
                     indicadores=None):
    """Señales de un activo con parámetros θ.

    Regresa un DataFrame con: los 3 votos, la suma de votos, s (fuerza en
    [−1, 1]), direccion = sign(s) y el ATR (para SL/TP).

    `indicadores` y `umbral` solo se cambian en el experimento de un solo
    indicador (p. ej. indicadores=["voto_rsi"], umbral=1). La estrategia
    oficial usa los 3 indicadores con umbral = 2.
    """
    votos = calcular_votos(ohlcv, params)
    if indicadores is None:
        indicadores = NOMBRES_VOTOS
    s = confirmar(votos[indicadores], umbral)
    salida = votos.copy()
    salida["suma_votos"] = votos[indicadores].sum(axis=1)
    salida["s"] = s
    salida["direccion"] = np.sign(s)
    salida["atr"] = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    return salida


def correlacion_votos(votos):
    """Matriz de correlación entre los 3 votos (redundancia de indicadores)."""
    return votos[NOMBRES_VOTOS].corr()
