# Checklist de cumplimiento — Lab 02 MyST · Equipo 2 · Nivel C

Cada requisito apunta al lugar del repositorio donde se cumple. Las cifras están en
`docs/resultados/` (generadas por `python main.py`).

## Parámetros fijos

- [x] Comisión 0.125% en entrada y salida — `src/config.py:COMISION`, `src/backtest.py:_Motor._tasa_costo`
- [x] Sin apalancamiento (Σ|w| ≤ 1) — `src/backtest.py:_Motor._ejecutar_ordenes`, `tests/test_restricciones.py`
- [x] Largos y cortos ejercidos — `docs/resultados/metricas_wf_oos.csv` (n_largas, n_cortas)
- [x] Capital $1,000,000 — `src/config.py:CAPITAL_INICIAL`
- [x] Confirmación 2 de 3 — `src/signals.py:confirmar`, `tests/test_confirmacion.py`
- [x] Objetivo Calmar — `src/optimize.py:correr_estudio`, `src/metrics.py:calmar`
- [x] SEED = 42 centralizada — `src/config.py:SEED`
- [x] Walk-forward 6 → 1 meses, paso mensual — `src/optimize.py:generar_ventanas`
- [x] Ventana de régimen de 63 días — `src/config.py:VENTANA_REGIMEN`
- [x] 100 trials por régimen por ventana — `src/config.py:N_TRIALS`, `docs/resultados/optimizacion_resumen.json`
- [x] 252 días, Rf = 0 — `src/config.py`

## Estructura y entorno

- [x] Estructura obligatoria, toda la lógica en `src/`, notebook sin lógica — `notebooks/analysis.ipynb`
- [x] `requirements.txt` con versiones fijadas (`==`), sin librerías de análisis técnico
- [x] `.gitignore` (.venv, .cache, __pycache__, checkpoints); `data/` y `docs/` versionados
- [x] `main.py` con `if __name__ == "__main__":` y `--stage`/`--quick`

## Datos

- [x] Descarga diaria con `auto_adjust=True` y datos congelados — `data/prices_daily.csv`, `data/download_metadata.json`
- [x] Auditoría — `docs/resultados/auditoria_datos.csv`
- [x] Limpieza sin forward-fill — `src/data.py:clean_data`
- [x] Split 80/20 cronológico — `docs/resultados/splits.json`

## Estrategia, motor y métricas

- [x] 3 indicadores de 3 familias + ATR, a mano — `src/signals.py`
- [x] Causalidad (t → t+1) — `tests/test_causalidad.py`, `tests/test_restricciones.py::test_ejecucion_en_t_mas_1`
- [x] Correlación entre señales — `docs/resultados/correlacion_senales.csv`
- [x] Motor event-driven, SL primero, gaps al open, cortos, log de operaciones — `src/backtest.py`, `docs/resultados/operaciones_wf_oos.csv`
- [x] Golden file de contabilidad — `tests/test_contabilidad.py`
- [x] Métricas (Sharpe, Sortino, Calmar, MDD, win rate, PF, etc.) y tablas mensual/trimestral/anual — `src/metrics.py`, `docs/resultados/retornos_*.csv`

## Régimen (Nivel B adaptado a diario)

- [x] K-means K = 3, re-etiquetado explícito, fit solo con datos ≤ train — `src/regimes.py`, `tests/test_regimen_causal.py`
- [x] Actualización semanal y persistencia de 2 actualizaciones — `src/regimes.py:clasificar`
- [x] Validación: persistencia, matriz de transición, silhouette, bootstrap y estabilidad TRAIN vs. TEST — `docs/resultados/regimenes_validacion.json`, `regimen_persistencia_*.csv`, `metricas_por_regimen_*.csv`

## Optimización

- [x] TPE con 30 trials aleatorios iniciales, mínimo de operaciones, fallback declarado — `src/optimize.py`
- [x] θ robusto (mediana del top 10%) — `src/optimize.py:theta_robusto`, `docs/resultados/parametros_por_ventana.csv`
- [x] Variantes por activo y compartida, y rolling vs. anchored — `docs/resultados/comparacion_variantes.csv`
- [x] Purga y embargo — `src/optimize.py:preparar_muestra`, `backtest_activo`
- [x] WFE — `docs/resultados/comparacion_variantes.csv`, `wf_is_por_ventana.csv`
- [x] Diagnósticos de Optuna — `docs/figures/14*.png`

## Portafolio (Nivel C)

- [x] σ_p, MRC, RC, Euler — `src/portfolio.py`, `tests/test_risk_parity.py`
- [x] EW, RP naive y RP Spinu con verificación < 1e-4 — `src/portfolio.py:pesos_rp_spinu`
- [x] Σ muestral de 504 días + comparación con EWMA y Ledoit-Wolf — `docs/resultados/estimadores_sigma.csv`
- [x] Composición w^target = m·w̃/max(1, Σ|w̃|) y conflictos — `src/portfolio.py:pesos_objetivo`
- [x] Turnover con drift, disparador híbrido y barrido f × δ — `docs/resultados/rebalanceo_barrido.csv`
- [x] Benchmarks: EW, naive, Buy & Hold y activos individuales — `docs/resultados/metricas_wf_oos.csv`

## TEST y robustez

- [x] θ congelado con SHA-256 y candado git — `docs/resultados/theta_congelado.json`, `main.py:etapa_test`
- [ ] TEST real corrido una sola vez — `docs/resultados/test_lock.json` (lo corre el equipo después de los commits)
- [x] 2 de 3 vs. un indicador — `docs/resultados/un_indicador.csv`
- [x] Sensibilidad ±20% — `docs/resultados/sensibilidad.csv`
- [x] Curva de costos y equilibrio — `docs/resultados/curva_costos.csv`, `costos_resumen.json`
- [x] Costos realistas y slippage — `docs/resultados/costos_realistas_train.csv`, `slippage_train.csv`

## Entregables

- [x] README con integrantes, instalación, comando, semilla, datos, tiempo y uso de IA
- [x] `docs/borrador_reporte.md` → `docs/reporte.pdf`
- [x] `docs/borrador_presentacion.md` → `docs/presentacion.pdf` (≤ 12 diapositivas + portada + cierre)
- [ ] Rellenar la fecha de exposición en el README
- [ ] Regenerar los PDFs después del TEST (`python main.py --stage report`)
