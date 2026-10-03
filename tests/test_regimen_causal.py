"""Pruebas del régimen: causalidad de las features y las etiquetas, y re-etiquetado estable."""
import numpy as np
import pandas as pd

from src import config, regimes


def test_features_causales(panel):
    """Las features en t son las mismas con toda la serie que con datos solo hasta t."""
    completas = regimes.calcular_features(panel)
    t = 1200
    hasta_t = {campo: tabla.iloc[:t + 1] for campo, tabla in panel.items()}
    parciales = regimes.calcular_features(hasta_t)
    pd.testing.assert_series_equal(completas.iloc[t], parciales.iloc[t], check_names=False)


def test_etiqueta_no_cambia_con_datos_posteriores(panel):
    """Agregar datos después de t no cambia la etiqueta de régimen en t ni las anteriores."""
    features = regimes.calcular_features(panel)
    fechas = features.index
    horario = [(fechas[801], regimes.ajustar_modelo(features, fechas[800]))]
    completa = regimes.clasificar(features, horario)
    for t in [900, 1503, 2500, len(fechas) - 1]:
        hasta_t = regimes.clasificar(features.iloc[:t + 1], horario)
        assert hasta_t.iloc[t] == completa.iloc[t]
        assert (hasta_t.to_numpy() == completa.iloc[:t + 1].to_numpy()).all()


def test_modelo_solo_usa_datos_hasta_fin_train(panel):
    """Los datos posteriores a fin_train no influyen en los centroides del K-means."""
    features = regimes.calcular_features(panel)
    fin = features.index[1000]
    con_todo = regimes.ajustar_modelo(features, fin)
    recortado = regimes.ajustar_modelo(features.iloc[:1001], fin)
    np.testing.assert_allclose(con_todo.kmeans.cluster_centers_,
                               recortado.kmeans.cluster_centers_)


def test_reetiquetado_invariante_a_permutaciones():
    """El nombre de cada régimen no depende del orden en que K-means numera los clusters."""
    columnas = config.FEATURES_MODELO
    centroides = np.array([[0.0, 0.0, 1.5],     # volatilidad baja, R² alto -> Tendencia
                           [2.5, 2.0, -0.5],    # volatilidad alta -> Crisis
                           [0.3, 0.4, -1.0]])   # el resto -> Reversión
    base = regimes.etiquetar_centroides(centroides, columnas)
    assert base == {0: config.TENDENCIA, 1: config.CRISIS, 2: config.REVERSION}
    for permutacion in [[1, 2, 0], [2, 0, 1], [0, 2, 1]]:
        mapeo = regimes.etiquetar_centroides(centroides[permutacion], columnas)
        for nuevo_id, viejo_id in enumerate(permutacion):
            assert mapeo[nuevo_id] == base[viejo_id]


def test_filtro_de_persistencia():
    """Un cambio de régimen se confirma solo si aparece en 2 actualizaciones seguidas."""
    class ModeloFalso:
        features = ["x"]

        def predecir(self, x):
            return np.array([int(x[0])])

    crudo = [0] * 5 + [2] * 5 + [0] * 10 + [2] * 10 + [0] * 5  # el primer 2 dura 1 actualización
    fechas = pd.bdate_range("2020-01-01", periods=len(crudo))
    features = pd.DataFrame({"x": crudo}, index=fechas)
    serie = regimes.clasificar(features, [(fechas[0], ModeloFalso())], cada=5)
    assert (serie.iloc[:20] == 0).all()     # el 2 aislado no se confirma
    assert (serie.iloc[25:30] == 2).all()   # con 2 seguidas, sí
