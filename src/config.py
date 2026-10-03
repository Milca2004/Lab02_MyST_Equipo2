"""
Aquí viven todos los números del proyecto: constantes, semilla y rutas.
Ningún otro archivo escribe números "a mano" los importan de este.
"""
import random
from pathlib import Path

# Reproducibilidad

SEED = 42                         # semilla fija: los resultados salen igual cada vez


TICKERS = ["NVDA", "AMZN", "TSLA", "META", "NFLX", "GOOGL"]   
INTEGRANTES = ["Arturo", "Milca", "Paula"]  


def asignar_activos(tickers=TICKERS, integrantes=INTEGRANTES, seed=SEED):
    """Reparte las 6 acciones entre los 3 integrantes, 2 a cada uno.

    Cómo funciona:
        1. Se fija la semilla 42, así el sorteo siempre da lo mismo.
        2. Se mezcla el orden de las 6 acciones.
        3. Se van tomando de 2 en 2, en orden alfabético de integrantes.
    Resultado: Arturo = GOOGL, NVDA; Milca = NFLX, TSLA; Paula = AMZN, META.
    """
    random.seed(seed)
    orden = random.sample(tickers, len(tickers))
    asignacion = {}
    for k, nombre in enumerate(integrantes):
        asignacion[nombre] = orden[2 * k: 2 * k + 2]
    return asignacion


ASIGNACION = asignar_activos()

# Datos

FECHA_INICIO = "2015-01-01"
FECHA_FIN = "2026-08-31"          # este día sí se incluye
FRACCION_TRAIN = 0.80             # el primer 80% se usa para entrenar (TRAIN), el último 20% para la prueba final (TEST)
UMBRAL_RET_EXTREMO = 0.25         # si un día el precio sube o baja más de 25%, se anota en la auditoría

# Parámetros fijos del laboratorio (no se optimizan)
CAPITAL_INICIAL = 1_000_000.0
COMISION = 0.00125                # 0.125% del monto de cada operación, al abrir y al cerrar
DIAS_ANIO = 252                   # días de bolsa en un año, para anualizar las métricas
RF = 0.0                          # tasa libre de riesgo
UMBRAL_CONFIRMACION = 2           # la suma de votos (sin importar el signo) debe ser al menos 2 para abrir
PISO_MDD = 0.01                   # en el Calmar, el drawdown máximo cuenta como mínimo 1%

# Indicadores
VENTANA_ATR = 14                  # el ATR siempre usa 14 días (suavizado de Wilder)

# Régimen (K-means sobre un índice con las 6 acciones al mismo peso)

VENTANA_REGIMEN = 63              # 3 meses de días hábiles
MA_RAPIDA_REGIMEN = 21            # medias móviles que miden qué tan fuerte es la tendencia
MA_LENTA_REGIMEN = 63
N_REGIMENES = 3
# Variables que usa el K-means. Se eligieron viendo solo los datos de TRAIN:
# con estas 3 el silhouette (qué tan separados quedan los grupos) fue 0.448,
# contra 0.315 con las 5. Las otras 2 se descartaron porque ρ₁ con 63 datos es
# casi puro ruido (error estándar de 1/√63, o sea 0.126) y la fuerza de
# tendencia repite lo que ya dice R². Las 5 se siguen calculando para las figuras.
FEATURES_MODELO = ["volatilidad", "atr_norm", "r2"]
ACTUALIZAR_REGIMEN_CADA = 5       # el régimen se recalcula cada 5 días hábiles (una semana)
PERSISTENCIA_REGIMEN = 2          # un cambio de régimen se acepta hasta que sale 2 veces seguidas
MIN_HISTORIA_REGIMEN = 252        # días mínimos de historia para entrenar el K-means
REGIMENES = ["Tendencia", "Reversión", "Crisis"]
TENDENCIA, REVERSION, CRISIS = 0, 1, 2   # el número que identifica a cada régimen
N_BOOTSTRAP = 1000                # cuántos remuestreos se hacen en el bootstrap

# Multiplicador de exposición según el régimen: qué tanto se invierte
M_REGIMEN = {TENDENCIA: 1.0, REVERSION: 0.7, CRISIS: 0.3}

# Optimización y walk-forward

N_TRIALS = 100                    # pruebas por régimen en cada ventana (el PDF pide entre 100 y 200)
N_STARTUP_TRIALS = 30             # pruebas al azar antes de que Optuna (TPE) empiece a elegir con criterio
MESES_TRAIN_WF = 6                # cada ventana entrena con 6 meses
MESES_TEST_WF = 1                 # y se prueba en el mes siguiente
DIAS_CALENTAMIENTO = 252          # días de historia antes del primer entrenamiento
DIAS_EMBARGO = 5                  # los últimos 5 días del train se dejan fuera del cálculo del objetivo
# Mínimo de operaciones. El lineamiento sugería 10 y 5, pero con datos diarios
# y la regla 2 de 3 solo el 1% de los casos (ventana, activo) llega a 10
# operaciones en 6 meses con CUALQUIER combinación de parámetros θ (el 72%
# llega a 5). Por eso se bajó a 5 y 3 (más o menos 1 operación cada 5 semanas).
# Decisión acordada por el equipo.
N_MIN_GLOBAL = 5                  # operaciones mínimas por activo en 6 meses
N_MIN_REGIMEN = 3                 # operaciones mínimas por régimen
MIN_DIAS_REGIMEN = 20             # días mínimos que debe tener un régimen en el train
FRACCION_ROBUSTA = 0.10           # se toma el 10% mejor de las pruebas válidas y de ahí la mediana
OBJETIVO_INVALIDO = -1e9          # puntaje pésimo que se da a una prueba que no cumple las restricciones

# Portafolio (Risk Parity)

VENTANA_COV = 504                 # 2 años de rendimientos diarios para estimar la covarianza
MIN_DIAS_COV = 126                # días mínimos para estimar Σ al principio
LAMBDA_EWMA = 0.94                # qué tanto pesa lo reciente en el estimador EWMA
TOL_RP = 1e-4                     # tolerancia: ningún activo se aleja de su parte justa (1/n) del riesgo por más de esto
UMBRAL_CORR_CONFLICTO = 0.7       # correlación desde la cual se revisan las señales en conflicto entre activos
FRECUENCIAS_REBALANCEO = {"diario": 1, "semanal": 5, "quincenal": 10, "mensual": 21}   # cada cuántos días hábiles
DELTAS_REBALANCEO = [0.02, 0.05, 0.10, 0.20]   # tamaños de banda que se prueban en el barrido
FRECUENCIA_DEFAULT = "semanal"
DELTA_DEFAULT = 0.10

# Escenario de costos realistas (sección 13)
SPREAD_BPS_REALISTA = 2.0         # spread completo en puntos base; se paga la mitad en cada lado
BORROW_FEE_REALISTA = 0.005       # 0.5% al año sobre el monto en corto
IMPACTO_ETA = 0.1                 # costo de impacto = eta·|q/ADV|^(2/3), como fracción del precio
VENTANA_ADV = 20                  # días para promediar el volumen diario (ADV)


# Rutas
RAIZ = Path(__file__).resolve().parents[1]
DIR_DATA = RAIZ / "data"
ARCHIVO_PRECIOS = DIR_DATA / "prices_daily.csv"
ARCHIVO_METADATA = DIR_DATA / "download_metadata.json"
DIR_DOCS = RAIZ / "docs"
DIR_CACHE = RAIZ / ".cache"


def rutas_salida(quick=False):
    """Devuelve las carpetas donde se guarda todo: (resultados, figuras, caché).

    En modo --quick todo va a .cache/quick/ y nunca se toca docs/.
    Si las carpetas no existen, las crea.
    """
    if quick:
        base = DIR_CACHE / "quick"
        dirs = (base / "resultados", base / "figures", base / "cache")
    else:
        dirs = (DIR_DOCS / "resultados", DIR_DOCS / "figures", DIR_CACHE / "full")
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs
