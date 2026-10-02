"""Regla 2 de 3: s_i = (1/3)·Σ votos si |Σ votos| ≥ 2; 0 en otro caso."""
import pandas as pd
import pytest

from src.signals import confirmar


def s_de(votos):
    tabla = pd.DataFrame([votos], columns=["voto_ema", "voto_rsi", "voto_bb"])
    return confirmar(tabla).iloc[0]


@pytest.mark.parametrize("signo", [1, -1])
def test_un_indicador_no_abre(signo):
    assert s_de([signo, 0, 0]) == 0
    assert s_de([0, 0, signo]) == 0


@pytest.mark.parametrize("signo", [1, -1])
def test_dos_a_favor_y_uno_neutral_abre(signo):
    assert s_de([signo, signo, 0]) == pytest.approx(signo * 2 / 3)
    assert s_de([0, signo, signo]) == pytest.approx(signo * 2 / 3)


@pytest.mark.parametrize("signo", [1, -1])
def test_tres_a_favor(signo):
    assert s_de([signo, signo, signo]) == pytest.approx(signo * 1.0)


@pytest.mark.parametrize("signo", [1, -1])
def test_caso_borde_dos_a_favor_uno_en_contra_no_abre(signo):
    # (+1, +1, −1): Σ = 1 < 2 -> no se abre (el tercero anula la confirmación)
    assert s_de([signo, signo, -signo]) == 0
    assert s_de([-signo, signo, signo]) == 0


def test_un_solo_indicador_con_umbral_1():
    tabla = pd.DataFrame({"voto_rsi": [1.0, 0.0, -1.0]})
    assert confirmar(tabla, umbral=1).tolist() == [1.0, 0.0, -1.0]
