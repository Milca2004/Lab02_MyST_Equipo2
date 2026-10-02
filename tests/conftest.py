"""Utilidades compartidas por las pruebas."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import DatosMotor  # noqa: E402


@pytest.fixture(scope="session")
def panel():
    from src import data
    return data.load_prices()


def motor_sintetico(open_, high, low, close, s, atr=1.0, m_sl=50.0, m_tp=50.0, max_hold=100,
                    capital=1_000_000.0):
    """DatosMotor de un activo con precios y señales fijados a mano."""
    T = len(open_)

    def col(x):
        return np.asarray(x, dtype=float).reshape(T, 1)

    return DatosMotor(
        fechas=pd.bdate_range("2020-01-01", periods=T), activos=["X"],
        open=col(open_), high=col(high), low=col(low), close=col(close), s=col(s),
        atr=np.full((T, 1), atr), m_sl=np.full((T, 1), m_sl), m_tp=np.full((T, 1), m_tp),
        max_hold=np.full((T, 1), max_hold), capital=capital)
