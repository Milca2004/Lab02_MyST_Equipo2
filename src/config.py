"""Configuración central del Lab 02 (MyST, ITESO) — Equipo 2, Nivel C.

Todas las constantes, la semilla y las rutas viven aquí. Ningún otro módulo
define números "mágicos": los importan de este archivo.
"""
import random
from pathlib import Path

# ----------------------------------------------------------------------------
# Reproducibilidad
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# Activos y asignación por integrante
# ----------------------------------------------------------------------------
TICKERS = ["NVDA", "AMZN", "TSLA", "META", "NFLX", "GOOGL"]
INTEGRANTES = ["Arturo", "Milca", "Paula"]  # orden alfabético


def asignar_activos(tickers=TICKERS, integrantes=INTEGRANTES, seed=SEED):
    """Reproduce la asignación aleatoria de activos.

    random.seed(42) -> random.sample(tickers, 6) -> pares consecutivos
    repartidos a los integrantes en orden alfabético.
    Resultado: Arturo = GOOGL, NVDA; Milca = NFLX, TSLA; Paula = AMZN, META.
    """
    random.seed(seed)
    orden = random.sample(tickers, len(tickers))
    asignacion = {}
    for k, nombre in enumerate(integrantes):
        asignacion[nombre] = orden[2 * k: 2 * k + 2]
    return asignacion


ASIGNACION = asignar_activos()

# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
FECHA_INICIO = "2015-01-01"
FECHA_FIN = "2026-08-31"          # inclusive
FRACCION_TRAIN = 0.80             # primer 80% = TRAIN, último 20% = TEST
UMBRAL_RET_EXTREMO = 0.25         # |r| > 25% se reporta en la auditoría

# ----------------------------------------------------------------------------
# Parámetros fijos del laboratorio (no se optimizan)
# ----------------------------------------------------------------------------
CAPITAL_INICIAL = 1_000_000.0
COMISION = 0.00125                # 0.125% sobre el nocional, entrada y salida
DIAS_ANIO = 252                   # anualización
RF = 0.0                          # tasa libre de riesgo
UMBRAL_CONFIRMACION = 2           # |suma de votos| >= 2 para abrir
PISO_MDD = 0.01                   # |MDD| mínimo de 1% en el Calmar

# ----------------------------------------------------------------------------
# Indicadores
# ----------------------------------------------------------------------------
VENTANA_ATR = 14                  # ATR con ventana fija (Wilder)

# ----------------------------------------------------------------------------
# Régimen (K-means sobre índice equiponderado)
# ----------------------------------------------------------------------------
VENTANA_REGIMEN = 63              # 3 meses hábiles
MA_RAPIDA_REGIMEN = 21            # medias para la fuerza de tendencia
MA_LENTA_REGIMEN = 63
N_REGIMENES = 3
# Subconjunto usado por el K-means (selección hecha solo con TRAIN):
# silhouette 0.448 vs 0.315 con las 5 features. ρ₁ con 63 obs. es casi puro
# ruido (error estándar ≈ 1/√63 ≈ 0.126) y la fuerza de tendencia es
# redundante con R². Las 5 se siguen calculando para las figuras.
FEATURES_MODELO = ["volatilidad", "atr_norm", "r2"]
ACTUALIZAR_REGIMEN_CADA = 5       # días hábiles (semanal)
PERSISTENCIA_REGIMEN = 2          # actualizaciones seguidas para confirmar
MIN_HISTORIA_REGIMEN = 252        # días mínimos para ajustar el K-means
REGIMENES = ["Tendencia", "Reversión", "Crisis"]
TENDENCIA, REVERSION, CRISIS = 0, 1, 2
N_BOOTSTRAP = 1000

# Multiplicador de exposición por régimen m(régimen)
M_REGIMEN = {TENDENCIA: 1.0, REVERSION: 0.7, CRISIS: 0.3}

# ----------------------------------------------------------------------------
# Optimización y walk-forward
# ----------------------------------------------------------------------------
N_TRIALS = 100                    # por régimen por ventana (rango 100–200)
N_STARTUP_TRIALS = 30             # trials aleatorios antes de TPE
MESES_TRAIN_WF = 6
MESES_TEST_WF = 1
DIAS_CALENTAMIENTO = 252          # historia previa al primer train del WF
DIAS_EMBARGO = 5                  # últimos días del train fuera del objetivo
# Mínimo de operaciones. El lineamiento sugería 10/5, pero con datos diarios y
# la regla 2 de 3 solo el 1% de los pares (ventana, activo) alcanza 10
# operaciones en 6 meses con CUALQUIER θ (72% alcanza 5). Se adaptó a 5/3
# (≈ 1 operación cada 5 semanas), decisión acordada por el equipo.
N_MIN_GLOBAL = 5                  # operaciones mínimas por activo en 6 meses
N_MIN_REGIMEN = 3                 # operaciones mínimas en estudio por régimen
MIN_DIAS_REGIMEN = 20             # días mínimos del régimen en el train
FRACCION_ROBUSTA = 0.10           # top 10% de trials válidos -> mediana
OBJETIVO_INVALIDO = -1e9

# ----------------------------------------------------------------------------
# Portafolio (Risk Parity)
# ----------------------------------------------------------------------------
VENTANA_COV = 504                 # 2 años de rendimientos diarios
MIN_DIAS_COV = 126                # mínimo para estimar Σ al inicio
LAMBDA_EWMA = 0.94
TOL_RP = 1e-4                     # max |RC_i/σ_p − 1/n|
UMBRAL_CORR_CONFLICTO = 0.7
FRECUENCIAS_REBALANCEO = {"diario": 1, "semanal": 5, "quincenal": 10, "mensual": 21}
DELTAS_REBALANCEO = [0.02, 0.05, 0.10, 0.20]
FRECUENCIA_DEFAULT = "semanal"
DELTA_DEFAULT = 0.10

# ----------------------------------------------------------------------------
# Escenario de costos realistas (sección 13)
# ----------------------------------------------------------------------------
SPREAD_BPS_REALISTA = 2.0         # spread completo; se paga la mitad por lado
BORROW_FEE_REALISTA = 0.005       # 0.5% anual sobre el nocional en corto
IMPACTO_ETA = 0.1                 # costo = eta·|q/ADV|^(2/3) (fracción del precio)
VENTANA_ADV = 20

# ----------------------------------------------------------------------------
# Rutas (relativas a la raíz del repositorio)
# ----------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATA = RAIZ / "data"
ARCHIVO_PRECIOS = DIR_DATA / "prices_daily.csv"
ARCHIVO_METADATA = DIR_DATA / "download_metadata.json"
DIR_DOCS = RAIZ / "docs"
DIR_CACHE = RAIZ / ".cache"


def rutas_salida(quick=False):
    """Regresa (dir_resultados, dir_figuras, dir_cache).

    En modo --quick todo se escribe en .cache/quick/ y nunca en docs/.
    """
    if quick:
        base = DIR_CACHE / "quick"
        dirs = (base / "resultados", base / "figures", base / "cache")
    else:
        dirs = (DIR_DOCS / "resultados", DIR_DOCS / "figures", DIR_CACHE / "full")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs
