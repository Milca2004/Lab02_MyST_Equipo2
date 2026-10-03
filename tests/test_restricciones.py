"""Restricciones: sin apalancamiento, SL primero, gaps al open y ejecución en t+1."""
import numpy as np
import pytest

from src import config, portfolio, regimes
from src.backtest import run_backtest
from tests.conftest import motor_sintetico


def simulacion_portafolio(panel):
    """Portafolio de 6 activos con θ agresivo (muchas operaciones) en 2018–2020."""
    features = regimes.calcular_features(panel)
    fechas = panel["close"].index
    modelo = regimes.ajustar_modelo(features, fechas[700])
    regimen = regimes.clasificar(features, [(fechas[701], modelo)])
    theta = {"ema_fast": 8, "ema_gap": 20, "rsi_window": 10, "rsi_lower": 40, "rsi_upper": 60,
             "bb_window": 15, "bb_k": 1.5, "m_sl": 1.5, "m_tp": 2.0, "max_hold": 8}
    thetas = {a: {j: theta for j in range(3)} for a in config.TICKERS}
    f_sim = fechas[750:1500]
    senales = portfolio.construir_senales(panel, regimen, [(f_sim[0], f_sim[-1], thetas)], f_sim)
    ctx = portfolio.ContextoSimulacion(panel, regimen, f_sim)
    return ctx.simular(senales, "rp", "diario", 0.02)


def test_nunca_apalancado(panel):
    res = simulacion_portafolio(panel)
    assert res.n_operaciones > 50
    assert res.exposicion_bruta_max <= 1.0 + 1e-9
    assert res.operaciones["direccion"].nunique() == 2          # largos y cortos


def test_sin_apalancamiento_un_activo_con_s_1():
    open_ = [100.0] * 5
    res = run_backtest(motor_sintetico(open_, [101] * 5, [99] * 5, open_, [1, 0, 0, 0, 0]))
    assert res.exposicion_bruta_max <= 1.0 + 1e-12


def test_sl_primero_si_sl_y_tp_en_la_misma_barra():
    # Entra largo a 100 en el día 1 con ATR 1, m_sl = m_tp = 2 -> SL 98, TP 102.
    # Día 2: low 97 y high 103 tocan ambos -> se ejecuta el SL a 98.
    open_ = [100, 100, 100, 100]
    high = [101, 101, 103, 101]
    low = [99, 99, 97, 99]
    close = [100, 100, 100, 100]
    res = run_backtest(motor_sintetico(open_, high, low, close, [1, 0, 0, 0],
                                       m_sl=2.0, m_tp=2.0))
    op = res.operaciones.iloc[0]
    assert op["motivo_salida"] == "SL" and op["precio_salida"] == 98


def test_gap_se_ejecuta_al_open():
    # Largo a 100, SL = 98. El día 2 abre en 95 (debajo del SL): sale a 95, no a 98.
    open_ = [100, 100, 95, 95]
    high = [101, 101, 96, 96]
    low = [99, 99, 94, 94]
    close = [100, 100, 95, 95]
    res = run_backtest(motor_sintetico(open_, high, low, close, [1, 0, 0, 0],
                                       m_sl=2.0, m_tp=5.0))
    op = res.operaciones.iloc[0]
    assert op["motivo_salida"] == "SL" and op["precio_salida"] == 95


def test_sl_primero_en_corto():
    # Corto a 100 en el día 1 con ATR 1, m_sl = m_tp = 2 -> SL 102 (arriba), TP 98 (abajo).
    # Día 2: high 103 y low 97 tocan ambos -> se ejecuta el SL a 102 (pérdida).
    open_ = [100, 100, 100, 100]
    high = [101, 101, 103, 101]
    low = [99, 99, 97, 99]
    close = [100, 100, 100, 100]
    res = run_backtest(motor_sintetico(open_, high, low, close, [-1, 0, 0, 0],
                                       m_sl=2.0, m_tp=2.0))
    op = res.operaciones.iloc[0]
    assert op["direccion"] == "corto"
    assert op["motivo_salida"] == "SL" and op["precio_salida"] == 102
    assert op["pnl_bruto"] < 0


def test_tp_en_corto():
    # Corto a 100, TP = 98. Día 2: low 97 toca solo el TP -> sale a 98 con ganancia.
    open_ = [100, 100, 100, 100]
    high = [101, 101, 101, 101]
    low = [99, 99, 97, 99]
    close = [100, 100, 98, 98]
    res = run_backtest(motor_sintetico(open_, high, low, close, [-1, 0, 0, 0],
                                       m_sl=2.0, m_tp=2.0))
    op = res.operaciones.iloc[0]
    assert op["motivo_salida"] == "TP" and op["precio_salida"] == 98
    assert op["pnl_bruto"] > 0


def test_gap_en_corto_se_ejecuta_al_open():
    # Corto a 100, SL = 102. El día 2 abre en 105 (arriba del SL): sale a 105, no a 102.
    open_ = [100, 100, 105, 105]
    high = [101, 101, 106, 106]
    low = [99, 99, 104, 104]
    close = [100, 100, 105, 105]
    res = run_backtest(motor_sintetico(open_, high, low, close, [-1, 0, 0, 0],
                                       m_sl=2.0, m_tp=5.0))
    op = res.operaciones.iloc[0]
    assert op["motivo_salida"] == "SL" and op["precio_salida"] == 105


def test_ejecucion_en_t_mas_1():
    # La señal sale al cierre del día 2; la entrada es al open del día 3 (precio 103).
    open_ = [100, 101, 102, 103, 104, 105]
    close = [100, 101, 102, 103, 104, 105]
    high = [c + 1 for c in close]
    low = [c - 1 for c in close]
    s = [0, 0, 1, 0, 0, 0]
    datos = motor_sintetico(open_, high, low, close, s, max_hold=100)
    res = run_backtest(datos)
    assert res.posiciones.iloc[2, 0] == 0                    # aún sin posición al cierre de t
    assert res.posiciones.iloc[3, 0] > 0                     # abierta en t+1
    assert res.efectivo.iloc[3] == pytest.approx(
        1_000_000 - res.posiciones.iloc[3, 0] * 103 * (1 + config.COMISION), abs=0.005)


def test_senal_cero_no_cierra():
    open_ = [100.0] * 8
    res = run_backtest(motor_sintetico(open_, [101] * 8, [99] * 8, open_, [1, 0, 0, 0, 0, 0, 0, 0]))
    assert len(res.operaciones) == 0 and res.posiciones.iloc[-1, 0] > 0


def test_reversa_cierra_y_abre():
    open_ = [100.0] * 6
    s = [1, 0, -1, 0, 0, 0]
    res = run_backtest(motor_sintetico(open_, [101] * 6, [99] * 6, open_, s))
    assert res.operaciones.iloc[0]["motivo_salida"] == "señal"
    assert res.posiciones.iloc[-1, 0] < 0
