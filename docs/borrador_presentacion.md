# Presentación — Lab 02 MyST · Equipo 2 · Nivel C

10 minutos de exposición + 5 de preguntas. Máximo 12 diapositivas (sin contar portada y cierre).

## 1. Problema, datos y activos — Milca (50 s)

- 6 mega-cap tech diarias, 2015-01-02 a 2026-08-31 (congelado)
- TRAIN hasta 2024-04-26 · TEST desde 2024-04-29
- Milca: NFLX, TSLA · Paula: AMZN, META · Arturo: GOOGL, NVDA
- Sesgo de supervivencia: B&H es casi imbatible en retorno → se juzga por Calmar
- Figura: `docs/figures/06a_regimenes_linea_tiempo.png`

## 2. Señal: 3 familias y regla 2 de 3 — Milca (50 s)

- EMA (tendencia) · RSI (momento) · Bollinger (volatilidad)
- s = (1/3)·Σ votos si |Σ| ≥ 2; (+1, +1, −1) NO abre
- Compuerta (¿hay posición?) ≠ fuerza (¿cuánto riesgo?)
- Figura: `docs/figures/13_correlacion_senales.png`

## 3. ¿Qué aporta 2 de 3? — Milca (55 s)

- 2 de 3: 286 operaciones, Calmar 0.04
- solo EMA: 1,508 ops, Calmar 0.13
- solo RSI: 706 ops, Calmar -0.14
- solo Bollinger: 850 ops, Calmar -0.17
- Figura: `docs/figures/09_dos_de_tres_vs_un_indicador.png`

## 4. Motor de backtest — Paula (50 s)

- Ejecución en t+1 al open · SL primero si SL y TP en la misma barra
- Gap → sale al open · comisión 0.125% por lado · cortos con pasivo
- Sin apalancamiento: Σ|w| ≤ 1 verificado en cada ejecución
- Figura: `docs/figures/11_costos_vs_bruto_wf_oos.png`

## 5. Optimización y walk-forward — Paula (60 s)

- Optuna TPE, 100 trials por régimen por ventana, objetivo Calmar
- 95 ventanas 6→1 meses · 416,300 configuraciones
- WFE rendimiento 0.02 · WFE Calmar 0.00
- Figura: `docs/figures/14d_optuna_superficie_3d.png`

## 6. WF-OOS vs. benchmarks — Paula (60 s)

- RP: CAGR 0.4%, MDD 10.8%, Calmar 0.04
- EW: Calmar 0.08 · B&H: CAGR 44.6%, Calmar 0.72
- Exposición media 9.9%: w = w^RP·s·m
- Figura: `docs/figures/01_valor_portafolio.png`

## 7. Sensibilidad ±20% — Paula (50 s)

- Calmar base (θ congelado, TRAIN) 0.13
- Uno a la vez, a todos los activos y regímenes
- Figura: `docs/figures/04_sensibilidad.png`

## 8. ¿A qué costo deja de ser rentable? — Paula (45 s)

- Equilibrio: 0.183% por lado
- Margen de seguridad 1.47× frente a 0.125%
- Figura: `docs/figures/05_curva_costos.png`

## 9. Regímenes (K-means, K = 3) — Arturo (55 s)

- Features: volatilidad, atr_norm, r2 · silhouette 0.448
- Actualización semanal + persistencia de 2 actualizaciones
- Duración promedio: Tendencia 51 d, Reversión 74 d, Crisis 83 d
- Figura: `docs/figures/06c_valor_con_regimenes.png`

## 10. Risk Parity: la ilusión del 50/50 — Arturo (55 s)

- Con pesos iguales TSLA + NVDA = 41.3% del riesgo
- Spinu: convexo, solución única, RC_i/σ_p = 1/6 ± 1e-4
- Figura: `docs/figures/07a_contribuciones_riesgo.png`

## 11. Rebalanceo: banda + calendario — Arturo (50 s)

- Elegido en TRAIN: diario, δ = 0.2
- Turnover contra el peso post-drift (no contra el objetivo viejo)
- Figura: `docs/figures/07d_barrido_rebalanceo.png`

## 12. TEST: una sola vez — Arturo (55 s)

- Sistema congelado: CAGR 1.3%, MDD 8.7%, Calmar 0.14
- Buy & Hold: CAGR 28.2%, Calmar 0.98
- Se tocó UNA vez, con candado (hash de θ + commit)
- Figura: `docs/figures/01_valor_portafolio.png`

## Cierre — los 3 (30 s)

- WF-OOS: Calmar 0.04 (EW 0.08), MDD 10.8%, exposición media 9.9%
- WFE 0.02: Sobrevive alrededor de 2% del rendimiento in-sample: por debajo de 0.5, el resultado in-sample es mayormente ruido ajustado.
- Equilibrio de costos en 0.183% por lado (margen 1.47×)
- TEST congelado: Calmar 0.14

Tiempo total de exposición: 665 s ≈ 11.1 min (+ portada).