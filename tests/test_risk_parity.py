"""Pruebas de Risk Parity: Spinu, Euler, naive, composición de pesos, turnover y conflictos."""
import numpy as np
import pandas as pd
import pytest

from src import portfolio
from src.signals import confirmar


def cov_desde(sigma, corr):
    """Matriz de covarianzas a partir de volatilidades y correlaciones."""
    sigma = np.asarray(sigma, dtype=float)
    return np.outer(sigma, sigma) * np.asarray(corr, dtype=float)


def test_spinu_iguala_contribuciones_sintetica():
    """Con una Σ inventada, todos los activos aportan 1/3 del riesgo (tolerancia 1e-4)."""
    corr = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
    cov = cov_desde([0.10, 0.25, 0.40], corr)
    w = portfolio.pesos_rp_spinu(cov)
    rc = portfolio.risk_contribution_pct(w, cov)
    assert np.max(np.abs(rc - 1 / 3)) < 1e-4
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()


def test_spinu_iguala_contribuciones_datos_reales(panel):
    """Con la Σ muestral de 504 días de los 6 activos, cada uno aporta 1/6 del riesgo."""
    rend = panel["close"].pct_change().iloc[1:505].to_numpy()
    cov = portfolio.cov_muestral(rend)
    w = portfolio.pesos_rp_spinu(cov)
    assert np.max(np.abs(portfolio.risk_contribution_pct(w, cov) - 1 / 6)) < 1e-4


def test_euler_suma_exacta():
    """Por el teorema de Euler, la suma de las contribuciones es exactamente σ_p."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T / 100
    w = rng.dirichlet(np.ones(6))
    assert portfolio.risk_contribution(w, cov).sum() == pytest.approx(
        portfolio.portfolio_vol(w, cov), abs=1e-10)


def test_naive_igual_a_rp_con_correlaciones_identicas():
    """Si todas las correlaciones son iguales, volatilidad inversa coincide con Risk Parity."""
    n = 5
    corr = np.full((n, n), 0.5) + 0.5 * np.eye(n)
    cov = cov_desde([0.1, 0.2, 0.3, 0.4, 0.5], corr)
    np.testing.assert_allclose(portfolio.pesos_naive(cov), portfolio.pesos_rp_spinu(cov), atol=1e-4)


def test_ejemplo_notas_ilusion_50_50():
    """Ejemplo de las notas: con σ = (5%, 25%) y ρ = 0, el 50/50 no es balanceado en riesgo."""
    cov = cov_desde([0.05, 0.25], np.eye(2))
    w = np.array([0.5, 0.5])
    varianza_activo2 = w[1] ** 2 * cov[1, 1] / (w @ cov @ w)
    assert varianza_activo2 == pytest.approx(0.9615, abs=1e-4)  # el activo 2 aporta ≈ 96.15%
    np.testing.assert_allclose(portfolio.pesos_naive(cov), [0.8333, 0.1667], atol=1e-4)


def test_composicion_paso_7_ejemplo_integrador():
    """Ejemplo de las notas: w^RP = (0.5, 0.3, 0.2), tres activos con votos distintos, m = 0.7."""
    votos = pd.DataFrame([[1, 1, 1], [1, 1, 0], [1, -1, -1]], dtype=float)  # activos A, B, C
    s = confirmar(votos).to_numpy()
    np.testing.assert_allclose(s, [1.0, 2 / 3, 0.0])
    w_target, _ = portfolio.pesos_objetivo([0.5, 0.3, 0.2], s, 0.7)
    np.testing.assert_allclose(w_target, [0.35, 0.14, 0.0], atol=1e-12)


def test_nunca_apalancado_en_composicion():
    """Aunque los pesos de largo plazo sumen más de 1, la exposición bruta no pasa de 1."""
    w_target, _ = portfolio.pesos_objetivo([0.6, 0.6, 0.6], [1, -1, 1], 1.0)
    assert np.abs(w_target).sum() <= 1.0 + 1e-12


def test_turnover_contra_peso_post_drift():
    """El turnover se mide contra el peso después del drift, no contra el objetivo viejo.

    Ejemplo del equipo: objetivo viejo (0.5, 0.5) y rendimientos (+10%, -10%) dejan el
    portafolio en (0.55, 0.45). Con objetivo nuevo (0.6, 0.4): contra el objetivo viejo
    T = 0.10 (sobreestima); contra el peso post-drift T = 0.05, que es lo que se opera.
    """
    w_post = portfolio.pesos_despues_drift([0.5, 0.5], [0.10, -0.10])
    np.testing.assert_allclose(w_post, [0.55, 0.45])
    assert portfolio.turnover([0.6, 0.4], [0.5, 0.5]) == pytest.approx(0.10)
    assert portfolio.turnover([0.6, 0.4], w_post) == pytest.approx(0.05)


def test_politica_de_conflictos():
    """Con señales opuestas y correlación alta: gana la más fuerte; si empatan, se reducen a la mitad."""
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    s, n = portfolio.resolver_conflictos([1.0, -2 / 3], corr)
    np.testing.assert_allclose(s, [1.0, 0.0])
    assert n == 1
    s, _ = portfolio.resolver_conflictos([2 / 3, -2 / 3], corr)
    np.testing.assert_allclose(s, [1 / 3, -1 / 3])
    s, n = portfolio.resolver_conflictos([1.0, -1.0], np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert n == 0  # correlación baja: no hay conflicto
