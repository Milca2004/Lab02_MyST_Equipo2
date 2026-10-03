# Lab02_MyST_Equipo2 — Estrategias de Trading con Análisis Técnico

Microestructuras y Sistemas de Trading · ITESO · **Nivel de alcance: C**

## Integrantes y activos

| Integrante | Activos | Módulos que revisa y commitea |
|---|---|---|
| Milca | NFLX, TSLA | `src/data.py`, `src/signals.py`, `tests/test_causalidad.py`, `tests/test_confirmacion.py` |
| Paula | AMZN, META | `src/backtest.py`, `src/metrics.py`, `src/optimize.py`, `tests/test_contabilidad.py`, `tests/test_restricciones.py`, `tests/test_metricas.py` (incluye sensibilidad y curva de costos) |
| Arturo | GOOGL, NVDA | `src/regimes.py`, `src/portfolio.py`, `tests/test_regimen_causal.py`, `tests/test_risk_parity.py` (incluye análisis de rebalanceo) |
| Los 3 | — | `src/config.py`, `src/plots.py`, `src/report.py`, `main.py`, README, notebook, reporte y presentación |

**Asignación de activos:** aleatoria y reproducible con `random.seed(42)` sobre
`['NVDA','AMZN','TSLA','META','NFLX','GOOGL']`: `random.sample(tickers, 6)` y luego pares
consecutivos asignados a los integrantes en orden alfabético (Arturo, Milca, Paula). Está en
`src/config.py:asignar_activos`.

## Descripción

Sistema de trading diario long/short sobre seis acciones tecnológicas de gran capitalización. Abre
posiciones solo cuando 2 de 3 indicadores implementados a mano (cruce de EMAs, RSI de Wilder y
Bandas de Bollinger) coinciden, con stop-loss y take-profit en múltiplos de ATR y un time-stop. Los
parámetros se optimizan con Optuna maximizando el Calmar por activo y por régimen de mercado
(K-means sobre un índice equiponderado), en un walk-forward de 6 meses de entrenamiento y 1 mes
fuera de muestra, con un motor de backtest event-driven que cobra 0.125% por lado y prohíbe el
apalancamiento. El tamaño de cada posición sale de Risk Parity (formulación convexa de Spinu)
multiplicado por la fuerza de la señal y por un factor de exposición por régimen, con rebalanceo
por calendario y banda. El último 20% de los datos (TEST) se evalúa una sola vez, con los
parámetros congelados y un candado que lo verifica.

## Instalación (Windows)

Python 3.12 (probado con 3.12.2; mínimo 3.10).

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Ejecución

```bat
python main.py
pytest -v
```

- `python main.py` corre todo: datos → auditoría → walk-forward (por activo, compartido y anchored)
  → regímenes → portafolio → robustez → test (con candado) → métricas → figuras → borradores → PDFs.
- `python main.py --stage {data,train,test,report,all}` corre una sola etapa.
- `python main.py --quick` es un ensayo rápido (20 trials, 4 ventanas; el "TEST" son los últimos
  2 meses de TRAIN). Escribe solo en `.cache/quick/`, nunca en `docs/`.
- El TEST solo corre con el working tree de git limpio. Con `python main.py` (etapa `all`) y
  cambios sin commitear, el TEST se omite con un aviso. Secuencia final:
  1. los tres commitean sus módulos;
  2. `python main.py --stage test`;
  3. `python main.py --stage report`;
  4. commit de `docs/`.
- Notebook: `jupyter nbconvert --to notebook --execute notebooks/analysis.ipynb`. No contiene
  lógica: solo lee `docs/resultados/` y llama a `src/`.
- Los checkpoints del walk-forward se guardan en `.cache/` (ignorado por git). Si cambia el código
  del motor, de las señales o de la optimización, borra `.cache/full/` antes de volver a correr.

## Semilla

`SEED = 42`, definida en `src/config.py`. Todas las semillas se derivan de ella: Optuna usa una
semilla determinística por (ventana, activo, régimen) y K-means y el bootstrap también parten de
`SEED`.

## Datos

- Fuente: Yahoo Finance vía `yfinance` 1.7.0, diario, `auto_adjust=True`, de 2015-01-01 a 2026-08-31.
- Fecha de descarga: **2026-10-01** (ver `data/download_metadata.json`).
- Datos congelados en `data/prices_daily.csv` (formato largo). `main.py` nunca descarga si el
  archivo existe.
- Los precios vienen ajustados por splits y dividendos (`auto_adjust=True`), por eso los valores
  históricos se ven más bajos que los que se cotizaban en su momento. Se ajustan para que un split
  no aparezca como una caída falsa en los retornos. Después de la descarga los datos no se modifican.

## Tiempo de ejecución aproximado

Medido en una Mac con 8 núcleos (7 procesos), con `.cache/` vacío:

- `python main.py` completo (sin TEST): **≈ 26 min**. El walk-forward rolling tarda ≈ 11 min, el
  anchored ≈ 15 min y los análisis, figuras y PDFs menos de 1 min.
- `python main.py --stage test`: unos minutos (re-optimiza mensualmente sobre TEST solo para la
  evaluación secundaria).
- `python main.py --quick`: ≈ 1 min. `pytest -v`: ≈ 6 s.
- Con menos núcleos el tiempo crece de forma aproximadamente proporcional. Los checkpoints de
  `.cache/` permiten retomar una corrida interrumpida.

## Fecha de exposición

Miércoles 7 de octubre de 2026. El último commit válido es a las 23:59 del martes 6 de octubre.

## Uso de asistencia de IA

El código, las pruebas, el notebook y los borradores del reporte y la presentación se generaron
con asistencia de IA (Claude Code, Anthropic), a partir de un prompt de especificación escrito por
el equipo. La IA no hizo commits. Cada integrante revisó línea por línea los módulos de la tabla de
arriba, los commiteó desde su propia cuenta y es responsable de poder explicarlos. Todas las cifras
del reporte y la presentación se leen de `docs/resultados/`, generado por `python main.py`; ninguna
se escribió a mano.
