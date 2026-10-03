"""Métricas: Sharpe, Sortino, MDD, Calmar y tablas, contra valores calculados a mano."""
import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.metrics import (cagr, calmar, duracion_max_drawdown, max_drawdown,
                         metricas_operaciones, returns_table, serie_drawdown, sharpe,
                         sortino)

RAIZ_252 = math.sqrt(config.DIAS_ANIO)


def curva(valores, inicio="2020-01-01"):
    """Serie de equity con fechas hábiles consecutivas."""
    return pd.Series(valores, index=pd.bdate_range(inicio, periods=len(valores)), dtype=float)


def curva_desde_rendimientos(rendimientos, v0=100.0):
    """V_0 = v0 y V_t = V_{t−1}·(1 + R_t)."""
    return curva(v0 * np.cumprod([1.0] + [1.0 + r for r in rendimientos]))


def test_sharpe_a_mano():
    # R = [1%, 2%, 3%]: media 2%, σ (ddof=1) 1% -> Sharpe = 2·√252
    equity = curva_desde_rendimientos([0.01, 0.02, 0.03])
    assert sharpe(equity) == pytest.approx(2.0 * RAIZ_252, rel=1e-9)


def test_sortino_a_mano():
    # R = [2%, −1%, 2%, −1%]: media 0.5%; σ_d = √((0.01² + 0.01²)/4) = √(5e−5)
    # Sortino = 0.005/√(5e−5)·√252 = √252/√2
    equity = curva_desde_rendimientos([0.02, -0.01, 0.02, -0.01])
    assert sortino(equity) == pytest.approx(RAIZ_252 / math.sqrt(2.0), rel=1e-9)


def test_drawdown_a_mano():
    # picos: 100, 120, 120, 120, 120, 130 -> peor caída 120 -> 60 = 50%
    equity = curva([100, 120, 90, 110, 60, 130])
    esperado = [0.0, 0.0, 0.25, 1 / 12, 0.5, 0.0]
    np.testing.assert_allclose(serie_drawdown(equity).to_numpy(), esperado)
    assert max_drawdown(equity) == pytest.approx(0.5)
    assert duracion_max_drawdown(equity) == 3       # 90, 110 y 60 bajo el pico de 120


def test_calmar_a_mano():
    # 252 rendimientos (1 año): cae a 90 el día 1 y sube sin caídas hasta 121.
    # CAGR = 121/100 − 1 = 21%, MDD = 10% -> Calmar = 2.1
    equity = curva([100.0] + list(np.linspace(90.0, 121.0, 252)))
    assert cagr(equity) == pytest.approx(0.21, rel=1e-9)
    assert max_drawdown(equity) == pytest.approx(0.10, rel=1e-9)
    assert calmar(equity) == pytest.approx(2.1, rel=1e-9)


def test_calmar_con_piso_de_mdd():
    # Curva que solo sube (MDD = 0): el MDD cuenta como 1% y el Calmar no explota.
    equity = curva(np.linspace(100.0, 110.0, 253))
    assert max_drawdown(equity) == 0.0
    assert calmar(equity) == pytest.approx(0.10 / config.PISO_MDD, rel=1e-9)


def test_curva_plana_no_divide_entre_cero():
    equity = curva([100.0] * 10)
    assert sharpe(equity) == 0.0
    assert sortino(equity) == 0.0
    assert max_drawdown(equity) == 0.0
    assert calmar(equity) == 0.0


def test_metricas_operaciones_a_mano():
    # PnL neto [100, −50, 200, −50]: 2 de 4 ganan; payoff = 150/50; PF = 300/100
    operaciones = pd.DataFrame({"pnl_neto": [100.0, -50.0, 200.0, -50.0],
                                "direccion": ["largo", "corto", "largo", "largo"]})
    m = metricas_operaciones(operaciones)
    assert m["n_operaciones"] == 4 and m["n_largas"] == 3 and m["n_cortas"] == 1
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["payoff"] == pytest.approx(3.0)
    assert m["profit_factor"] == pytest.approx(3.0)


def test_tabla_de_rendimientos_compone():
    # Cierres de mes 100 -> 110 -> 99: enero 0%, febrero +10%, marzo −10%,
    # y el trimestre compone: 1.10·0.90 − 1 = −1%.
    fechas = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    equity = pd.Series([100.0, 110.0, 99.0], index=fechas)
    tablas = returns_table(equity)
    np.testing.assert_allclose(tablas["mensual"].to_numpy(), [0.0, 0.10, -0.10])
    assert tablas["trimestral"].iloc[0] == pytest.approx(-0.01)
    assert tablas["anual"].iloc[0] == pytest.approx(-0.01)
