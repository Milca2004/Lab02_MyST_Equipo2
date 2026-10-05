# Presentación — Lab 02 MyST · Equipo 2 · Nivel C

10 minutos de exposición + 5 de preguntas. Máximo 12 diapositivas (sin contar portada y cierre).

## 1. Problema, datos y activos — Milca (40 s)

- 6 mega-cap tech diarias, 2015-01-02 a 2026-08-31 (congelado)
- TRAIN hasta 2024-04-26 · TEST desde 2024-04-29
- Milca: NFLX, TSLA · Paula: AMZN, META · Arturo: GOOGL, NVDA
- Sesgo de supervivencia: B&H casi imbatible en retorno → se juzga por Calmar
- Figura: `docs/figures/06a_regimenes_linea_tiempo.png`

## 2. Señal: 3 familias y regla 2 de 3 — Milca (40 s)

- EMA (tendencia) · RSI (momento) · Bollinger (volatilidad)
- s = (1/3)·Σ votos si |Σ| ≥ 2; (+1, +1, −1) NO abre
- Compuerta (¿hay posición?) ≠ fuerza (¿cuánto riesgo?)
- Figura: `docs/figures/13_correlacion_senales.png`

## 3. P1 · ¿Qué aporta 2 de 3? — Milca (50 s)

- 2 de 3: 286 operaciones, Calmar 0.04, MDD 10.8%
- solo EMA: 1,508 ops, Calmar 0.13, MDD 35.1%
- solo RSI: 706 ops, Calmar -0.14, MDD 42.6%
- solo Bollinger: 850 ops, Calmar -0.17, MDD 50.2%
- Filtra: menos operaciones que 100% de los indicadores solos; mejor Calmar que 67%
- Figura: `docs/figures/09_dos_de_tres_vs_un_indicador.png`

## 4. Motor de backtest y optimización — Milca (55 s)

- Ejecución en t+1 al open · SL primero si SL y TP en la misma barra · comisión 0.125%/lado
- Sin apalancamiento: Σ|w| ≤ 1 verificado en cada ejecución
- Optuna TPE, 100 trials por estudio, objetivo Calmar, 95 ventanas 6→1 meses
- 416,300 configuraciones evaluadas en TRAIN
- Figura: `docs/figures/14a_optuna_historia.png`

## 5. WF-OOS vs. benchmarks y degradación (P2) — Paula (50 s)

- RP: CAGR 0.4%, MDD 10.8%, Calmar 0.04 · exposición media 9.9%
- EW: Calmar 0.08 · B&H: CAGR 44.6%, Calmar 0.72
- P2 · CAGR WF-IS 14.1% → WF-OOS 0.3%: WFE 0.02 → sobrevive ≈ 2%
- WFE de Calmar 0.00 (< 0.5: lo in-sample es sobre todo ajuste)
- Figura: `docs/figures/01_valor_portafolio.png`

## 6. P3 · Sensibilidad ±20%: ¿meseta o pico? — Paula (50 s)

- Calmar base (θ congelado, TRAIN) 0.13 · rango -0.004 a 0.257
- Conserva el signo en 90% de los casos ±20%
- Dentro de ±50% de la base: 80%
- Parámetro más sensible: max_hold (rango de Calmar 0.19)
- Veredicto: meseta
- Figura: `docs/figures/04_sensibilidad.png`

## 7. P4 · ¿A qué costo deja de ser rentable? — Paula (45 s)

- Equilibrio: 0.183% por lado
- Margen de seguridad 1.47× frente a 0.125%
- 36.6 operaciones/año · costo 0.91% del capital al año
- Figura: `docs/figures/05_curva_costos.png`

## 8. Regímenes (K-means, K = 3) — Paula (40 s)

- Features: volatilidad, atr_norm, r2 · silhouette 0.448
- Actualización semanal + persistencia de 2 actualizaciones
- Duración promedio: Tendencia 51 d, Reversión 74 d, Crisis 83 d
- Figura: `docs/figures/06c_valor_con_regimenes.png`

## 9. P5 · ¿Difiere el desempeño entre regímenes? — Arturo (45 s)

- WF-OOS (pb/día, IC 95%): Tendencia -0.1 [-3.7, 3.5] · Reversión 0.9 [-1.1, 2.7] · Crisis -0.3 [-1.8, 0.9]
- TEST (pb/día, IC 95%): Tendencia -0.3 [-6.3, 6.0] · Reversión 1.5 [-2.5, 5.7] · Crisis 1.3 [-0.5, 3.5]
- 6 de 6 IC bootstrap incluyen 0 → no hay diferencia significativa
- Aporte de la capa: control de riesgo (m = 1.0/0.7/0.3); vol. anual WF-OOS Tendencia 6.9% · Reversión 4.1% · Crisis 2.8%
- Figura: `docs/figures/15_ic_por_regimen.png`

## 10. P6 · Risk Parity vs. pesos iguales (+ rebalanceo) — Arturo (55 s)

- WF-OOS · Calmar RP 0.04 vs. EW 0.08 vs. naive 0.01; MDD 10.8% vs. 10.8%
- TEST · Calmar RP 0.14 vs. EW 0.04 vs. naive 0.19; MDD 8.7% vs. 9.9% vs. 8.1%
- Riesgo: máx RC_i RP 16.7% vs. EW 21.0%; TSLA+NVDA 33.3% vs. 41.3%
- A costa de: CAGR WF-OOS 0.4% vs. 0.8%; más peso en GOOGL (21.7%), menos en TSLA (13.6%)
- Rebalanceo elegido en TRAIN: diario, δ = 0.2
- Figura: `docs/figures/07a_contribuciones_riesgo.png`

## 11. TEST: una sola vez — Arturo (45 s)

- Sistema congelado: CAGR 1.3%, MDD 8.7%, Calmar 0.14
- EW: Calmar 0.04 · Buy & Hold: CAGR 28.2%, Calmar 0.98
- P2 · Calmar WF-OOS 0.04 → TEST 0.14; CAGR 0.4% → 1.3%
- Se tocó UNA vez, con candado (hash de θ + commit d397929)
- Figura: `docs/figures/03_rendimientos_test.png`

## 12. P7 · Tres limitaciones para capital real — Arturo (45 s)

- 1) Universo: 6 tech correlacionadas (corr. media 0.47), elegidas ex post
- 2) Pocas operaciones (36.6/año) y 469,100 configuraciones probadas → Calmar con mucho error
- 3) Ejecución: llenado completo al open o en SL/TP, sin impacto ni préstamo de títulos
- ⚠ El backtest asume ejecución perfecta: los resultados son una cota optimista
- Realista en TEST (spread + borrow + impacto): Calmar 0.14 → 0.12; con slippage 20 bps → -0.09
- Figura: `docs/figures/11_costos_vs_bruto_test.png`

## Cierre — los 3 (30 s)

- WF-OOS: Calmar 0.04 (EW 0.08), MDD 10.8%, exposición media 9.9%
- WFE 0.02: sobrevive ≈ 2% del rendimiento in-sample
- Equilibrio de costos en 0.183% por lado (margen 1.47×)
- TEST congelado: Calmar 0.14

Tiempo total de exposición: 590 s ≈ 9.8 min (+ portada).
Reparto: Arturo 200 s · Milca 195 s · Paula 195 s (cierre de 30 s repartido entre los 3).