"""Detección de régimen de mercado (K-means, K = 3) sobre un índice equiponderado.

Causalidad:
- Las features en t usan solo una ventana móvil de 63 días que termina en t.
- El scaler y el K-means se ajustan solo con datos ≤ fin del train de cada
  ventana; después solo se hace `predict` con el modelo congelado.
- La clasificación se actualiza cada 5 días hábiles (rejilla fija sobre el
  calendario global) y un cambio se confirma solo si el nuevo régimen aparece
  en 2 actualizaciones seguidas. La etiqueta en t no cambia al agregar datos
  posteriores.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import config

FEATURES = ["volatilidad", "atr_norm", "fuerza_tendencia", "r2", "autocorr"]


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def indice_equiponderado(close):
    """Índice EW: r_t = (1/n)·Σ_i r_{i,t};  I_t = Π (1 + r_t)."""
    rend = close.pct_change().mean(axis=1).fillna(0.0)
    return (1.0 + rend).cumprod(), rend


def calcular_features(panel, ventana=config.VENTANA_REGIMEN):
    """Features de régimen (ventana móvil de 63 días, solo datos ≤ t).

    volatilidad:       σ_t = √(1/n·Σ r²_{t−i})  (magnitud del riesgo; separa crisis)
    atr_norm:          promedio de los 6 activos de mean_63(TR/close_{t−1})
                       (rango intradía; comparable entre activos y épocas)
    fuerza_tendencia:  |MA_21 − MA_63| (sobre log I) / σ_t  (tendencia en sigmas)
    r2:                R² de log I_t contra el tiempo en 63 días (linealidad)
    autocorr:          ρ₁ de los rendimientos del índice (persistencia vs. reversión)
    """
    close, high, low = panel["close"], panel["high"], panel["low"]
    indice, rend = indice_equiponderado(close)
    log_indice = np.log(indice)

    volatilidad = np.sqrt((rend ** 2).rolling(ventana).mean())

    cierre_previo = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - cierre_previo).abs(),
                                           (low - cierre_previo).abs()))
    atr_norm = (tr / cierre_previo).rolling(ventana).mean().mean(axis=1)

    ma_rapida = log_indice.rolling(config.MA_RAPIDA_REGIMEN).mean()
    ma_lenta = log_indice.rolling(config.MA_LENTA_REGIMEN).mean()
    fuerza = (ma_rapida - ma_lenta).abs() / volatilidad

    tiempo = pd.Series(np.arange(len(log_indice), dtype=float), index=log_indice.index)
    r2 = log_indice.rolling(ventana).corr(tiempo) ** 2   # R² = corr² en regresión simple

    autocorr = rend.rolling(ventana).corr(rend.shift(1))

    features = pd.DataFrame({"volatilidad": volatilidad, "atr_norm": atr_norm,
                             "fuerza_tendencia": fuerza, "r2": r2, "autocorr": autocorr})
    return features


# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
@dataclass
class ModeloRegimen:
    scaler: StandardScaler
    kmeans: KMeans
    mapeo: dict            # id de cluster -> régimen (0 Tendencia, 1 Reversión, 2 Crisis)
    features: list
    fin_train: pd.Timestamp
    silhouette: float
    centroides: pd.DataFrame

    def predecir(self, x):
        """Régimen de una o varias filas de features (sin re-ajustar nada)."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        ids = self.kmeans.predict(self.scaler.transform(x))
        return np.array([self.mapeo[int(c)] for c in ids])


def etiquetar_centroides(centroides, features):
    """Re-etiquetado explícito (resuelve el label switching).

    1. Crisis = cluster con mayor volatilidad en su centroide.
    2. De los otros dos, Tendencia = mayor (fuerza_tendencia + r2) en unidades
       estandarizadas (si alguna no está en el subconjunto, se usa la que esté).
    3. El restante = Reversión.
    Solo depende de los valores de los centroides, no de los IDs de cluster.
    """
    tabla = pd.DataFrame(centroides, columns=features)
    id_crisis = int(tabla["volatilidad"].idxmax())
    resto = [i for i in tabla.index if i != id_crisis]
    columnas_tendencia = [f for f in ["fuerza_tendencia", "r2"] if f in features]
    puntaje = tabla.loc[resto, columnas_tendencia].sum(axis=1)
    id_tendencia = int(puntaje.idxmax())
    id_reversion = [i for i in resto if i != id_tendencia][0]
    return {id_crisis: config.CRISIS, id_tendencia: config.TENDENCIA,
            id_reversion: config.REVERSION}


def ajustar_modelo(features, fin_train, columnas=config.FEATURES_MODELO, seed=config.SEED):
    """Ajusta scaler + K-means con las features de fechas ≤ fin_train.

    La historia es creciente (todo lo disponible ≤ fin_train): el régimen de
    Crisis necesita haber visto episodios de crisis para existir como cluster.
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


# ----------------------------------------------------------------------------
# Clasificación causal con filtro de persistencia
# ----------------------------------------------------------------------------
def clasificar(features, horario, cada=config.ACTUALIZAR_REGIMEN_CADA,
               persistencia=config.PERSISTENCIA_REGIMEN):
    """Serie de régimen confirmado (−1 donde aún no hay modelo vigente).

    horario: lista ordenada de (fecha_desde, ModeloRegimen); a partir de
    fecha_desde se usa ese modelo (p. ej. el mes k+1 usa el modelo de la ventana k).

    - Fechas de actualización: posición en el calendario global múltiplo de
      `cada`, más el primer día con modelo vigente (inicialización).
    - Filtro: el régimen confirmado cambia solo si el mismo régimen nuevo sale
      en `persistencia` actualizaciones seguidas.
    """
    fechas = features.index
    regimen = np.full(len(fechas), -1, dtype=int)
    confirmado, candidato, repeticiones = -1, -1, 0
    k_modelo = -1
    for t, fecha in enumerate(fechas):
        while k_modelo + 1 < len(horario) and horario[k_modelo + 1][0] <= fecha:
            k_modelo += 1
        if k_modelo < 0:
            continue
        modelo = horario[k_modelo][1]
        x = features.iloc[t][modelo.features]
        es_actualizacion = (t % cada == 0) or confirmado == -1
        if es_actualizacion and x.notna().all():
            crudo = int(modelo.predecir(x.to_numpy())[0])
            if confirmado == -1:
                confirmado = crudo
            elif crudo == confirmado:
                candidato, repeticiones = -1, 0
            elif crudo == candidato:
                repeticiones += 1
                if repeticiones >= persistencia:
                    confirmado, candidato, repeticiones = crudo, -1, 0
            else:
                candidato, repeticiones = crudo, 1
                if repeticiones >= persistencia:
                    confirmado, candidato, repeticiones = crudo, -1, 0
        regimen[t] = confirmado
    return pd.Series(regimen, index=fechas, name="regimen")


# ----------------------------------------------------------------------------
# Validación
# ----------------------------------------------------------------------------
def duraciones(regimen):
    """Rachas consecutivas de cada régimen: DataFrame (regimen, inicio, fin, dias)."""
    regimen = regimen[regimen >= 0]
    rachas = []
    inicio = 0
    valores = regimen.to_numpy()
    for t in range(1, len(valores) + 1):
        if t == len(valores) or valores[t] != valores[inicio]:
            rachas.append({"regimen": int(valores[inicio]), "inicio": regimen.index[inicio],
                           "fin": regimen.index[t - 1], "dias": t - inicio})
            inicio = t
    return pd.DataFrame(rachas, columns=["regimen", "inicio", "fin", "dias"])


def matriz_transicion(regimen):
    """A_ij = P(S_{t+1} = j | S_t = i) diaria y duración esperada E[D_j] = 1/(1 − A_jj)."""
    valores = regimen[regimen >= 0].to_numpy()
    k = config.N_REGIMENES
    conteo = np.zeros((k, k))
    for a, b in zip(valores[:-1], valores[1:]):
        conteo[a, b] += 1
    filas = conteo.sum(axis=1, keepdims=True)
    A = np.divide(conteo, filas, out=np.zeros_like(conteo), where=filas > 0)
    duracion_esperada = [1.0 / (1.0 - A[j, j]) if A[j, j] < 1 else np.inf for j in range(k)]
    return A, duracion_esperada


def resumen_persistencia(regimen):
    """Duración promedio observada, # de rachas y proporción del tiempo por régimen."""
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
    """Media de rendimientos diarios con IC al 95% por bootstrap (remuestreo i.i.d.)."""
    rend = np.asarray(rend, dtype=float)
    rend = rend[np.isfinite(rend)]
    if len(rend) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    medias = np.empty(n)
    for b in range(n):
        muestra = rng.choice(rend, size=len(rend), replace=True)
        medias[b] = muestra.mean()
    return float(rend.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def metricas_por_regimen(equity, regimen, nombre=""):
    """Métricas de una curva separadas por el régimen vigente cada día.

    Para cada régimen: días, media diaria con IC 95% por bootstrap, volatilidad
    y Sharpe anualizados y MDD de la curva formada solo con esos días.
    El rendimiento de t se asigna al régimen confirmado al cierre de t−1
    (el que estaba vigente cuando se tomó la posición).
    """
    rend = equity.pct_change().dropna()
    regimen_previo = regimen.shift(1).reindex(rend.index)
    filas = []
    for j, etiqueta in enumerate(config.REGIMENES):
        r = rend[regimen_previo == j]
        media, inf, sup = bootstrap_media(r.to_numpy())
        sigma = r.std(ddof=1) if len(r) > 1 else np.nan
        curva = (1.0 + r).cumprod()
        mdd = float(((curva.cummax() - curva) / curva.cummax()).max()) if len(r) else np.nan
        filas.append({
            "nombre": nombre, "regimen": etiqueta, "dias": int(len(r)),
            "media_diaria": media, "ic95_inf": inf, "ic95_sup": sup,
            "ic_incluye_cero": bool(inf <= 0 <= sup) if np.isfinite(media) else None,
            "retorno_anualizado": media * config.DIAS_ANIO if np.isfinite(media) else np.nan,
            "volatilidad_anual": sigma * np.sqrt(config.DIAS_ANIO) if np.isfinite(sigma) else np.nan,
            "sharpe": media / sigma * np.sqrt(config.DIAS_ANIO) if sigma and np.isfinite(sigma) else np.nan,
            "mdd": mdd,
        })
    return pd.DataFrame(filas)


def correlacion_por_regimen(close, regimen):
    """Correlación promedio entre pares y matriz de correlación por régimen."""
    rend = close.pct_change().dropna()
    regimen = regimen.reindex(rend.index)
    matrices, filas = {}, []
    for j, etiqueta in enumerate(config.REGIMENES):
        r = rend[regimen == j]
        if len(r) < 10:
            continue
        corr = r.corr()
        matrices[etiqueta] = corr
        n = corr.shape[0]
        fuera_diagonal = corr.to_numpy()[~np.eye(n, dtype=bool)]
        filas.append({"regimen": etiqueta, "dias": len(r),
                      "correlacion_media": float(fuera_diagonal.mean())})
    return pd.DataFrame(filas), matrices
