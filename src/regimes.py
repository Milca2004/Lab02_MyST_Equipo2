"""


Detección de régimen de mercado con K-means (K = 3) sobre un índice equiponderado.
Todo es causal: las features en t usan solo una ventana de 63 días que termina en t, el
scaler y el K-means se ajustan con datos hasta el fin del train y después solo se predice.
El régimen se actualiza cada 5 días hábiles y un cambio se confirma únicamente si el nuevo
régimen aparece en 2 actualizaciones seguidas.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

FEATURES = ["volatilidad", "atr_norm", "fuerza_tendencia", "r2", "autocorr"]


# --- Features -----------------------------------------------------------------
def indice_equiponderado(close):
    """Índice con el rendimiento promedio de todos los activos. Devuelve (índice, rendimientos)."""
    rend = close.pct_change().mean(axis=1).fillna(0.0)
    return (1.0 + rend).cumprod(), rend


def calcular_features(panel, ventana=config.VENTANA_REGIMEN):
    """Features de régimen sobre ventana móvil (solo datos hasta t).

    volatilidad:      raíz del promedio de r² (separa las crisis)
    atr_norm:         rango verdadero promedio / cierre previo, promediado entre activos
    fuerza_tendencia: |MA rápida - MA lenta| del log del índice, en unidades de volatilidad
    r2:               qué tan lineal es el log del índice contra el tiempo
    autocorr:         autocorrelación de orden 1 (persistencia vs. reversión)
    """
    close, high, low = panel["close"], panel["high"], panel["low"]
    indice, rend = indice_equiponderado(close)
    log_indice = np.log(indice)

    volatilidad = np.sqrt((rend ** 2).rolling(ventana).mean())

    previo = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - previo).abs(), (low - previo).abs()))
    atr_norm = (tr / previo).rolling(ventana).mean().mean(axis=1)

    ma_rapida = log_indice.rolling(config.MA_RAPIDA_REGIMEN).mean()
    ma_lenta = log_indice.rolling(config.MA_LENTA_REGIMEN).mean()
    fuerza = (ma_rapida - ma_lenta).abs() / volatilidad

    tiempo = pd.Series(np.arange(len(log_indice), dtype=float), index=log_indice.index)
    r2 = log_indice.rolling(ventana).corr(tiempo) ** 2  # en regresión simple R² = corr²

    autocorr = rend.rolling(ventana).corr(rend.shift(1))

    return pd.DataFrame({"volatilidad": volatilidad, "atr_norm": atr_norm,
                         "fuerza_tendencia": fuerza, "r2": r2, "autocorr": autocorr})


# --- Modelo -------------------------------------------------------------------
@dataclass
class ModeloRegimen:
    scaler: StandardScaler
    kmeans: KMeans
    mapeo: dict            # id de cluster -> régimen (Tendencia, Reversión, Crisis)
    features: list
    fin_train: pd.Timestamp
    silhouette: float
    centroides: pd.DataFrame

    def predecir(self, x):
        """Régimen de una o varias filas de features, sin reajustar nada."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        ids = self.kmeans.predict(self.scaler.transform(x))
        return np.array([self.mapeo[int(c)] for c in ids])


def etiquetar_centroides(centroides, features):
    """Asigna nombre a cada cluster usando solo los valores de sus centroides.

    Crisis es el de mayor volatilidad. De los otros dos, Tendencia es el de mayor
    fuerza_tendencia + r2, y el que queda es Reversión. Así los ids del K-means no importan.
    """
    tabla = pd.DataFrame(centroides, columns=features)
    id_crisis = int(tabla["volatilidad"].idxmax())
    resto = [i for i in tabla.index if i != id_crisis]
    columnas = [f for f in ["fuerza_tendencia", "r2"] if f in features]
    id_tendencia = int(tabla.loc[resto, columnas].sum(axis=1).idxmax())
    id_reversion = next(i for i in resto if i != id_tendencia)
    return {id_crisis: config.CRISIS, id_tendencia: config.TENDENCIA,
            id_reversion: config.REVERSION}


def ajustar_modelo(features, fin_train, columnas=config.FEATURES_MODELO, seed=config.SEED):
    """Ajusta scaler y K-means con todo el historial hasta fin_train.

    Se usa toda la historia (no solo la ventana reciente) para que el cluster de crisis
    llegue a existir.
    """
    datos = features.loc[:fin_train, columnas].dropna()
    scaler = StandardScaler().fit(datos.to_numpy())
    x = scaler.transform(datos.to_numpy())
    kmeans = KMeans(n_clusters=config.N_REGIMENES, init="k-means++", n_init=10,
                    random_state=seed).fit(x)
    mapeo = etiquetar_centroides(kmeans.cluster_centers_, columnas)
    sil = float(silhouette_score(x, kmeans.labels_, random_state=seed,
                                 sample_size=min(len(x), 3000)))
    centroides = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=columnas)
    centroides["regimen"] = [config.REGIMENES[mapeo[i]] for i in range(config.N_REGIMENES)]
    return ModeloRegimen(scaler, kmeans, mapeo, list(columnas), pd.Timestamp(fin_train),
                         sil, centroides)


# --- Clasificación causal -----------------------------------------------------
def clasificar(features, horario, cada=config.ACTUALIZAR_REGIMEN_CADA,
               persistencia=config.PERSISTENCIA_REGIMEN):
    """Serie de régimen confirmado (-1 mientras no haya modelo vigente).

    horario: lista ordenada de (fecha_desde, ModeloRegimen). Desde esa fecha se usa ese
    modelo (por ejemplo, el mes k+1 usa el modelo ajustado en la ventana k).
    Se reclasifica cada `cada` días (y el primer día con modelo). El régimen confirmado solo
    cambia si el mismo régimen nuevo sale `persistencia` veces seguidas.
    """
    fechas = features.index
    regimen = np.full(len(fechas), -1, dtype=int)
    confirmado, candidato, repeticiones = -1, -1, 0
    k = -1
    for t, fecha in enumerate(fechas):
        while k + 1 < len(horario) and horario[k + 1][0] <= fecha:
            k += 1
        if k < 0:
            continue
        modelo = horario[k][1]
        x = features.iloc[t][modelo.features]
        if (t % cada == 0 or confirmado == -1) and x.notna().all():
            crudo = int(modelo.predecir(x.to_numpy())[0])
            if confirmado == -1:
                confirmado = crudo
            elif crudo == confirmado:
                candidato, repeticiones = -1, 0
            else:
                repeticiones = repeticiones + 1 if crudo == candidato else 1
                candidato = crudo
                if repeticiones >= persistencia:
                    confirmado, candidato, repeticiones = crudo, -1, 0
        regimen[t] = confirmado
    return pd.Series(regimen, index=fechas, name="regimen")


# --- Validación ---------------------------------------------------------------
def duraciones(regimen):
    """Rachas consecutivas de cada régimen: DataFrame (regimen, inicio, fin, dias)."""
    regimen = regimen[regimen >= 0]
    valores = regimen.to_numpy()
    rachas, inicio = [], 0
    for t in range(1, len(valores) + 1):
        if t == len(valores) or valores[t] != valores[inicio]:
            rachas.append({"regimen": int(valores[inicio]), "inicio": regimen.index[inicio],
                           "fin": regimen.index[t - 1], "dias": t - inicio})
            inicio = t
    return pd.DataFrame(rachas, columns=["regimen", "inicio", "fin", "dias"])


def matriz_transicion(regimen):
    """Matriz A_ij = P(régimen j mañana | régimen i hoy) y duración esperada 1 / (1 - A_jj)."""
    valores = regimen[regimen >= 0].to_numpy()
    k = config.N_REGIMENES
    conteo = np.zeros((k, k))
    for a, b in zip(valores[:-1], valores[1:]):
        conteo[a, b] += 1
    filas = conteo.sum(axis=1, keepdims=True)
    A = np.divide(conteo, filas, out=np.zeros_like(conteo), where=filas > 0)
    esperada = [1.0 / (1.0 - A[j, j]) if A[j, j] < 1 else np.inf for j in range(k)]
    return A, esperada


def resumen_persistencia(regimen):
    """Por régimen: proporción del tiempo, número de rachas y duración promedio."""
    rachas = duraciones(regimen)
    A, esperada = matriz_transicion(regimen)
    validos = regimen[regimen >= 0]
    filas = []
    for j, nombre in enumerate(config.REGIMENES):
        r = rachas[rachas["regimen"] == j]
        filas.append({"regimen": nombre,
                      "proporcion_tiempo": float((validos == j).mean()) if len(validos) else np.nan,
                      "n_rachas": int(len(r)),
                      "duracion_promedio_obs": float(r["dias"].mean()) if len(r) else np.nan,
                      "duracion_esperada_markov": float(esperada[j]),
                      "A_jj": float(A[j, j])})
    return pd.DataFrame(filas), A


def bootstrap_media(rend, n=config.N_BOOTSTRAP, seed=config.SEED):
    """Media de rendimientos con intervalo de 95% por bootstrap. Devuelve (media, inf, sup)."""
    rend = np.asarray(rend, dtype=float)
    rend = rend[np.isfinite(rend)]
    if len(rend) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    medias = [rng.choice(rend, size=len(rend), replace=True).mean() for _ in range(n)]
    return float(rend.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def metricas_por_regimen(equity, regimen, nombre=""):
    """Métricas de una curva de capital separadas por régimen.

    El rendimiento de t se asigna al régimen confirmado al cierre de t-1, que es el que
    estaba vigente cuando se tomó la posición.
    """
    rend = equity.pct_change().dropna()
    regimen_previo = regimen.shift(1).reindex(rend.index)
    anual = config.DIAS_ANIO
    filas = []
    for j, etiqueta in enumerate(config.REGIMENES):
        r = rend[regimen_previo == j]
        media, inf, sup = bootstrap_media(r.to_numpy())
        sigma = r.std(ddof=1) if len(r) > 1 else np.nan
        curva = (1.0 + r).cumprod()
        mdd = float(((curva.cummax() - curva) / curva.cummax()).max()) if len(r) else np.nan
        hay_media = np.isfinite(media)
        filas.append({
            "nombre": nombre, "regimen": etiqueta, "dias": int(len(r)),
            "media_diaria": media, "ic95_inf": inf, "ic95_sup": sup,
            "ic_incluye_cero": bool(inf <= 0 <= sup) if hay_media else None,
            "retorno_anualizado": media * anual if hay_media else np.nan,
            "volatilidad_anual": sigma * np.sqrt(anual) if np.isfinite(sigma) else np.nan,
            "sharpe": media / sigma * np.sqrt(anual) if sigma and np.isfinite(sigma) else np.nan,
            "mdd": mdd,
        })
    return pd.DataFrame(filas)


def correlacion_por_regimen(close, regimen):
    """Correlación promedio entre pares de activos y matriz de correlación por régimen."""
    rend = close.pct_change().dropna()
    regimen = regimen.reindex(rend.index)
    matrices, filas = {}, []
    for j, etiqueta in enumerate(config.REGIMENES):
        r = rend[regimen == j]
        if len(r) < 10:
            continue
        corr = r.corr()
        matrices[etiqueta] = corr
        fuera_diagonal = corr.to_numpy()[~np.eye(corr.shape[0], dtype=bool)]
        filas.append({"regimen": etiqueta, "dias": len(r),
                      "correlacion_media": float(fuera_diagonal.mean())})
    return pd.DataFrame(filas), matrices
