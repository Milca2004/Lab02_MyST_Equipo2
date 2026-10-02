"""Régimen: causalidad de features y etiquetas, e invarianza del re-etiquetado."""
import numpy as np
import pandas as pd

from src import config, regimes


def test_features_causales(panel):
    completas = regimes.calcular_features(panel)
    t = 1200
    truncado = {c: tabla.iloc[:t + 1] for c, tabla in panel.items()}
    parciales = regimes.calcular_features(truncado)
    pd.testing.assert_series_equal(completas.iloc[t], parciales.iloc[t], check_names=False)


def test_etiqueta_no_cambia_con_datos_posteriores(panel):
    features = regimes.calcular_features(panel)
    fechas = features.index
    modelo = regimes.ajustar_modelo(features, fechas[800])
    horario = [(fechas[801], modelo)]
    completa = regimes.clasificar(features, horario)
    for t in [900, 1503, 2500, len(fechas) - 1]:
        truncada = regimes.clasificar(features.iloc[:t + 1], horario)
        assert truncada.iloc[t] == completa.iloc[t]
        assert (truncada.to_numpy() == completa.iloc[:t + 1].to_numpy()).all()


def test_modelo_solo_usa_datos_hasta_fin_train(panel):
    features = regimes.calcular_features(panel)
    fin = features.index[1000]
    a = regimes.ajustar_modelo(features, fin)
    b = regimes.ajustar_modelo(features.iloc[:1001], fin)
    np.testing.assert_allclose(a.kmeans.cluster_centers_, b.kmeans.cluster_centers_)


def test_reetiquetado_invariante_a_permutaciones():
    columnas = config.FEATURES_MODELO
    centroides = np.array([[0.0, 0.0, 1.5],     # baja vol, alta R² -> Tendencia
                           [2.5, 2.0, -0.5],    # alta vol -> Crisis
                           [0.3, 0.4, -1.0]])   # resto -> Reversión
    base = regimes.etiquetar_centroides(centroides, columnas)
    assert base == {0: config.TENDENCIA, 1: config.CRISIS, 2: config.REVERSION}
    for permutacion in [[1, 2, 0], [2, 0, 1], [0, 2, 1]]:
        permutados = centroides[permutacion]
        mapeo = regimes.etiquetar_centroides(permutados, columnas)
        for nuevo_id, viejo_id in enumerate(permutacion):
            assert mapeo[nuevo_id] == base[viejo_id]


def test_filtro_de_persistencia():
    """Un cambio se confirma solo si aparece en 2 actualizaciones seguidas."""
    class ModeloFalso:
        features = ["x"]

        def predecir(self, x):
            return np.array([int(x[0])])

    crudo = [0] * 5 + [2] * 5 + [0] * 10 + [2] * 10 + [0] * 5   # 1 actualización de 2 aislada
    features = pd.DataFrame({"x": crudo}, index=pd.bdate_range("2020-01-01", periods=len(crudo)))
    serie = regimes.clasificar(features, [(features.index[0], ModeloFalso())], cada=5)
    assert (serie.iloc[:20] == 0).all()                 # el 2 aislado no se confirma
    assert (serie.iloc[25:30] == 2).all()               # 2 actualizaciones seguidas -> sí
