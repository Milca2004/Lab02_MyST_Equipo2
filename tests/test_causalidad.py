"""La señal en t calculada con datos hasta t es igual a la calculada con toda la serie."""
import numpy as np
import pandas as pd
import pytest

from src import data, signals
from src.optimize import CacheIndicadores

PARAMS = {"ema_fast": 12, "ema_gap": 40, "rsi_window": 14, "rsi_lower": 35, "rsi_upper": 65,
          "bb_window": 20, "bb_k": 1.8, "m_sl": 2.0, "m_tp": 3.0, "max_hold": 20}


@pytest.mark.parametrize("activo", ["TSLA", "NFLX"])
def test_senal_no_usa_futuro(panel, activo):
    ohlcv = data.asset_frame(panel, activo)
    completa = signals.generate_signals(ohlcv, PARAMS)
    n = len(ohlcv)
    for t in [300, 1500, n - 2]:            # uno cerca del final
        truncada = signals.generate_signals(ohlcv.iloc[:t + 1], PARAMS)
        fila_completa = completa.iloc[t]
        fila_truncada = truncada.iloc[t]
        pd.testing.assert_series_equal(fila_completa, fila_truncada, check_names=False)


def test_indicadores_individuales_causales(panel):
    close = data.asset_frame(panel, "TSLA")["close"]
    t = 1000
    for funcion in [lambda c: signals.ema(c, 20), lambda c: signals.rsi(c, 14),
                    lambda c: signals.bollinger(c, 20, 2.0)[1]]:
        assert np.isclose(funcion(close).iloc[t], funcion(close.iloc[:t + 1]).iloc[t])


def test_cache_igual_a_generate_signals(panel):
    """El caché del optimizador da exactamente lo mismo que signals.generate_signals."""
    ohlcv = data.asset_frame(panel, "META")
    cache = CacheIndicadores(ohlcv)
    for params in [PARAMS, dict(PARAMS, ema_fast=5, ema_gap=10, bb_k=3.0, rsi_window=7)]:
        a = signals.generate_signals(ohlcv, params)
        b = cache.senales(params)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
