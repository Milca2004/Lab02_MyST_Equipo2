"""Contabilidad del motor: golden file calculado a mano, equity y costos."""
import numpy as np
import pytest

from src import config
from src.backtest import Costos, run_backtest
from tests.conftest import motor_sintetico

C = config.COMISION
CAPITAL = 1_000_000.0


def test_golden_largo():
    # señal +1 al cierre del día 0 -> compra al open del día 1 a 100.
    # max_hold = 2 -> al cierre del día 2 lleva 2 días -> vende al open del día 3 a 110.
    open_ = [99, 100, 105, 110, 110]
    close = [99, 104, 108, 111, 110]
    high = [p + 1 for p in close]
    low = [min(o, c) - 1 for o, c in zip(open_, close)]
    s = [1, 0, 0, 0, 0]
    res = run_backtest(motor_sintetico(open_, high, low, close, s, max_hold=2))

    # a mano: sin apalancamiento, el nocional máximo es Capital/(1 + c)
    q = CAPITAL / (1 + C) / 100.0
    costo_entrada = q * 100 * C
    costo_salida = q * 110 * C
    pnl_bruto = q * (110 - 100)
    pnl_neto = pnl_bruto - costo_entrada - costo_salida
    efectivo_final = CAPITAL + pnl_neto

    op = res.operaciones.iloc[0]
    assert len(res.operaciones) == 1
    assert op["direccion"] == "largo" and op["motivo_salida"] == "time-stop"
    assert op["precio_entrada"] == 100 and op["precio_salida"] == 110
    assert op["pnl_bruto"] == pytest.approx(pnl_bruto, abs=0.005)
    assert op["costos"] == pytest.approx(costo_entrada + costo_salida, abs=0.005)
    assert op["pnl_neto"] == pytest.approx(pnl_neto, abs=0.005)
    assert res.efectivo.iloc[-1] == pytest.approx(efectivo_final, abs=0.005)
    assert res.efectivo.iloc[1] == pytest.approx(0.0, abs=0.005)   # todo invertido


def test_golden_corto():
    # señal −1 al cierre del día 0 -> vende en corto al open del día 1 a 200.
    # max_hold = 1 -> recompra al open del día 2 a 190.
    open_ = [201, 200, 190, 190]
    close = [200, 195, 190, 190]
    high = [max(o, c) + 1 for o, c in zip(open_, close)]
    low = [min(o, c) - 1 for o, c in zip(open_, close)]
    s = [-1, 0, 0, 0]
    res = run_backtest(motor_sintetico(open_, high, low, close, s, max_hold=1))

    q = CAPITAL / (1 + C) / 200.0
    efectivo_tras_corto = CAPITAL + q * 200 * (1 - C)            # Cash += q·P·(1 − c)
    pnl_neto = q * (200 - 190) - q * 200 * C - q * 190 * C
    op = res.operaciones.iloc[0]
    assert op["direccion"] == "corto"
    assert res.efectivo.iloc[1] == pytest.approx(efectivo_tras_corto, abs=0.005)
    # equity del día 1 = efectivo + q·P (q negativo) al cierre 195
    assert res.equity.iloc[1] == pytest.approx(efectivo_tras_corto - q * 195, abs=0.005)
    assert op["pnl_neto"] == pytest.approx(pnl_neto, abs=0.005)
    assert res.equity.iloc[-1] == pytest.approx(CAPITAL + pnl_neto, abs=0.005)


def test_equity_igual_efectivo_mas_posiciones(panel):
    from tests.test_restricciones import simulacion_portafolio
    res = simulacion_portafolio(panel)
    precios = panel["close"].loc[res.equity.index].to_numpy()
    reconstruido = res.efectivo.to_numpy() + (res.posiciones.to_numpy() * precios).sum(axis=1)
    np.testing.assert_allclose(res.equity.to_numpy(), reconstruido, rtol=1e-12)


def test_costos_son_comision_por_nocional(panel):
    from tests.test_restricciones import simulacion_portafolio
    res = simulacion_portafolio(panel)
    assert res.costos_totales == pytest.approx(C * res.nocional_operado.sum(), rel=1e-12)
    # PnL neto de las operaciones cerradas = bruto − costos
    ops = res.operaciones
    np.testing.assert_allclose(ops["pnl_neto"], ops["pnl_bruto"] - ops["costos"])


def test_costos_con_nocional_fijo():
    # 3 operaciones idénticas de nocional N: costo total = 3 × 2 × N × c (aprox., sin PnL)
    open_ = [100.0] * 12
    close = [100.0] * 12
    high = [101.0] * 12
    low = [99.0] * 12
    s = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
    res = run_backtest(motor_sintetico(open_, high, low, close, s, max_hold=1))
    assert len(res.operaciones) == 3
    nocionales = res.nocional_operado[res.nocional_operado > 0]
    assert res.costos_totales == pytest.approx(C * nocionales.sum(), rel=1e-12)
