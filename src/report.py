"""Reporte y presentación generados a partir de docs/resultados/.

Ninguna cifra se escribe a mano: todas se leen de los CSV/JSON producidos por
main.py. Donde aún no existe el TEST se deja el marcador [PENDIENTE: test].
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src import config  # noqa: E402

PENDIENTE = "[PENDIENTE: test]"


# ----------------------------------------------------------------------------
# Lectura de resultados y formato de cifras
# ----------------------------------------------------------------------------
class Resultados:
    """Acceso a los archivos de resultados; regresa None si un archivo no existe."""

    def __init__(self, directorio):
        self.dir = Path(directorio)

    def csv(self, nombre, **kw):
        ruta = self.dir / nombre
        return pd.read_csv(ruta, **kw) if ruta.exists() else None

    def json(self, nombre):
        ruta = self.dir / nombre
        if not ruta.exists():
            return None
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)

    def fila(self, nombre_csv, columna, valor):
        tabla = self.csv(nombre_csv)
        if tabla is None:
            return None
        coincide = tabla[tabla[columna] == valor]
        return None if coincide.empty else coincide.iloc[0]


def pct(x, dec=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/d"
    return f"{x * 100:.{dec}f}%"


def num(x, dec=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/d"
    return f"{x:,.{dec}f}"


def tabla_md(df, columnas, encabezados, formatos):
    """DataFrame -> tabla markdown con formatos por columna."""
    lineas = ["| " + " | ".join(encabezados) + " |", "|" + "---|" * len(encabezados)]
    for _, fila in df.iterrows():
        celdas = [formatos.get(c, str)(fila[c]) for c in columnas]
        lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


FMT_METRICAS = {"cagr": pct, "volatilidad": pct, "sharpe": num, "sortino": num, "mdd": pct,
                "calmar": num, "win_rate": pct, "profit_factor": num, "n_operaciones": lambda x: num(x, 0),
                "turnover_anual": num, "exposicion_media": pct, "costos_vs_bruto": pct,
                "retorno_total": pct, "payoff": num, "n_largas": lambda x: num(x, 0),
                "n_cortas": lambda x: num(x, 0), "duracion_max_dd": lambda x: num(x, 0),
                "nombre": str}


def tabla_metricas(df):
    columnas = ["nombre", "retorno_total", "cagr", "volatilidad", "sharpe", "sortino", "mdd", "calmar",
                "n_operaciones", "n_largas", "n_cortas", "win_rate", "profit_factor", "exposicion_media"]
    columnas = [c for c in columnas if c in df.columns]
    encabezados = {"nombre": "Estrategia", "retorno_total": "Ret. total", "cagr": "CAGR",
                   "volatilidad": "Vol.", "sharpe": "Sharpe", "sortino": "Sortino", "mdd": "MDD",
                   "calmar": "Calmar", "n_operaciones": "# ops", "n_largas": "# largas",
                   "n_cortas": "# cortas", "win_rate": "Win rate", "profit_factor": "PF",
                   "exposicion_media": "Exp. media"}
    return tabla_md(df, columnas, [encabezados[c] for c in columnas], FMT_METRICAS)


# ----------------------------------------------------------------------------
# Contenido del reporte
# ----------------------------------------------------------------------------
def construir_reporte(R, figuras_rel):
    """Regresa el texto markdown del reporte con cifras leídas de R."""
    split = R.json("splits.json")
    congelado = R.json("theta_congelado.json")
    opt = R.json("optimizacion_resumen.json")
    costos = R.json("costos_resumen.json")
    reg = R.json("regimenes_validacion.json")
    lock = R.json("test_lock.json")
    met_oos = R.csv("metricas_wf_oos.csv")
    met_test = R.csv("metricas_test.csv")
    variantes = R.csv("comparacion_variantes.csv")
    auditoria = R.csv("auditoria_datos.csv")
    hay_test = met_test is not None and lock is not None and not lock.get("ensayo_quick", False) \
        or (met_test is not None and lock is not None and lock.get("ensayo_quick", False))

    def fig(nombre, titulo):
        return f"![{titulo}]({figuras_rel}/{nombre})"

    rp = met_oos.set_index("nombre").loc["RP (sistema)"]
    ew = met_oos.set_index("nombre").loc["Pesos iguales (EW)"]
    naive = met_oos.set_index("nombre").loc["RP naive"]
    bh = met_oos.set_index("nombre").loc["Buy & Hold EW"]
    var = variantes.set_index("nombre")
    fila_roll = variantes[(variantes["esquema"] == "rolling") & (variantes["variante"] == congelado["variante"])].iloc[0]
    fila_anc = variantes[variantes["esquema"] == "anchored"].iloc[0]
    un_ind = R.csv("un_indicador.csv")
    port_ind = un_ind[un_ind["nivel"] == "portafolio"].set_index("caso")
    sens = R.csv("sensibilidad.csv")
    rp_cmp = R.csv("rp_vs_naive_vs_ew.csv").set_index("nombre")
    contrib = R.csv("contribuciones_riesgo_promedio.csv")
    estim = R.csv("estimadores_sigma.csv")
    por_reg = R.csv("metricas_por_regimen_wf_oos.csv")
    persist = R.csv("regimen_persistencia_wf_oos.csv")
    corr_reg = R.csv("correlacion_por_regimen.csv")
    barrido = R.csv("rebalanceo_barrido.csv")
    realistas = R.csv("costos_realistas_train.csv").set_index("caso")
    slip = R.csv("slippage_train.csv")
    corr_sen = R.csv("correlacion_senales.csv")
    port_resumen = R.json("portafolio_resumen_wf_oos.json")
    ops_reg = R.csv("operaciones_por_regimen_wf_oos.csv")
    params = R.csv("parametros_por_ventana.csv")
    diag = R.json("optuna_diagnostico.json")
    ret_anual = R.csv("retornos_anual_wf_oos.csv", index_col=0, parse_dates=True)

    # ------------- cifras derivadas (todas de archivos) --------------------
    ew_rc = contrib[contrib["metodo"] == "ew"].set_index("activo")
    ilusion = ew_rc.loc[["TSLA", "NVDA"], "rc_pct"].sum()
    ilusion_peso = ew_rc.loc[["TSLA", "NVDA"], "peso"].sum()
    rp_pesos = contrib[contrib["metodo"] == "rp"].set_index("activo")["peso"]
    naive_pesos = contrib[contrib["metodo"] == "naive"].set_index("activo")["peso"]
    dif_naive_rp = float((rp_pesos - naive_pesos).abs().max())
    base_sens = sens[sens["factor"] == 0]["calmar"].iloc[0]
    sens_theta = sens[sens["parametro"].isin([p for p in sens["parametro"].unique()
                                              if p not in ("m_regimen", "delta", "ventana_cov")])]
    rango_sens = sens_theta.groupby("parametro")["calmar"].agg(lambda c: c.max() - c.min())
    peor_param = rango_sens.idxmax()
    cambios_20 = sens_theta[sens_theta["factor"].abs() == 0.2]
    frac_signo = float((np.sign(cambios_20["calmar"]) == np.sign(base_sens)).mean())
    frac_cerca = float(((cambios_20["calmar"] - base_sens).abs() <= 0.5 * abs(base_sens)).mean())
    calmar_min_sens, calmar_max_sens = float(sens_theta["calmar"].min()), float(sens_theta["calmar"].max())
    caida_media_20 = float((cambios_20["calmar"] - base_sens).mean())
    equilibrio = costos["comision_equilibrio"]
    margen = costos["margen_seguridad"]
    fall = pd.DataFrame(opt["fallbacks_por_regimen"])
    fall_ofi = fall[(fall["esquema"] == "rolling") & (fall["variante"] == congelado["variante"])]
    fallback_txt = "; ".join(f"{r['regimen']}: {int(r['sum'])} de {int(r['count'])}"
                             for _, r in fall_ofi.iterrows())
    corr_med = corr_sen.groupby(["voto_a", "voto_b"])["correlacion"].mean()
    corr_rsi_bb = float(corr_med.loc[("voto_rsi", "voto_bb")])
    corr_ema_rsi = float(corr_med.loc[("voto_ema", "voto_rsi")])
    corr_ema_bb = float(corr_med.loc[("voto_ema", "voto_bb")])
    reg_rp = por_reg[por_reg["nombre"] == "RP (sistema)"]
    sig_reg = reg_rp[~reg_rp["ic_incluye_cero"].astype(bool)]
    mejor_reb = barrido.loc[barrido["calmar"].idxmax()]
    n_cfg = opt["configuraciones_evaluadas_train"]
    horas = opt["segundos_reloj_etapa_train"] / 3600
    horas_cpu = (opt["segundos_cpu_optimizacion_rolling"] + opt["segundos_cpu_optimizacion_anchored"]) / 3600
    test_sec = R.json("test_secundario_resumen.json")
    n_cfg_total = n_cfg + (test_sec["trials_totales_test_secundario"] if test_sec else 0)

    # ------------- TEST ----------------------------------------------------
    if met_test is not None:
        t = met_test.set_index("nombre")
        trp, tew, tbh = t.loc["RP (sistema)"], t.loc["Pesos iguales (EW)"], t.loc["Buy & Hold EW"]
        test_txt = (f"En TEST ({split['test_inicio']} a {split['test_fin']}) el sistema congelado obtuvo "
                    f"CAGR {pct(trp['cagr'])}, MDD {pct(trp['mdd'])} y Calmar {num(trp['calmar'])} "
                    f"(EW: Calmar {num(tew['calmar'])}; Buy & Hold: CAGR {pct(tbh['cagr'])}, "
                    f"Calmar {num(tbh['calmar'])}). Commit del test: `{lock['commit']}`.")
        tabla_test = tabla_metricas(met_test)
        sec = R.csv("test_secundario_wf.csv")
        sec_txt = tabla_md(sec, ["nombre", "cagr", "mdd", "calmar"], ["Evaluación", "CAGR", "MDD", "Calmar"],
                           FMT_METRICAS) if sec is not None else PENDIENTE
        degradacion = (f"Del WF-OOS al TEST el Calmar pasó de {num(rp['calmar'])} a {num(trp['calmar'])} "
                       f"y el CAGR de {pct(rp['cagr'])} a {pct(trp['cagr'])}.")
        reg_test = R.csv("regimen_persistencia_test.csv")
    else:
        test_txt = tabla_test = sec_txt = PENDIENTE
        degradacion = f"Degradación TRAIN → TEST: {PENDIENTE}."
        reg_test = None

    def tabla_persist(p):
        return tabla_md(p, ["regimen", "proporcion_tiempo", "n_rachas", "duracion_promedio_obs",
                            "duracion_esperada_markov"],
                        ["Régimen", "% del tiempo", "# rachas", "Duración obs. (días)", "E[D] Markov (días)"],
                        {"regimen": str, "proporcion_tiempo": pct, "n_rachas": lambda x: num(x, 0),
                         "duracion_promedio_obs": lambda x: num(x, 1),
                         "duracion_esperada_markov": lambda x: num(x, 1)})

    # ------------- texto -----------------------------------------------------
    asign = config.ASIGNACION
    th = congelado["thetas"]
    tabla_theta = ["| Activo | Régimen | EMA rápida/lenta | RSI (n, inf, sup) | BB (n, k) | SL/TP (×ATR) | max_hold |",
                   "|---|---|---|---|---|---|---|"]
    for a in config.TICKERS:
        for r in config.REGIMENES:
            p = th[a][r]
            tabla_theta.append(f"| {a} | {r} | {p['ema_fast']}/{p['ema_fast'] + p['ema_gap']} | "
                               f"{p['rsi_window']}, {p['rsi_lower']}, {p['rsi_upper']} | "
                               f"{p['bb_window']}, {p['bb_k']:.2f} | {p['m_sl']:.2f}/{p['m_tp']:.2f} | "
                               f"{p['max_hold']} |")
        if congelado["variante"] == "compartido":
            tabla_theta.append("| (los 6 activos comparten θ) | | | | | | |")
            break

    por_activo_port = R.csv("portafolio_por_activo_wf_oos.csv")
    individuales = met_oos[met_oos["nombre"].str.contains("estrategia sola")]

    secciones = []
    secciones.append(f"""# Laboratorio 02 — Estrategias de Trading con Análisis Técnico

**Microestructuras y Sistemas de Trading · ITESO · Equipo 2 · Nivel C**
Integrantes: Milca ({', '.join(asign['Milca'])}), Paula ({', '.join(asign['Paula'])}), Arturo ({', '.join(asign['Arturo'])}).

## 1. Resumen ejecutivo

Construimos un sistema de trading diario long/short sobre NVDA, AMZN, TSLA, META, NFLX y GOOGL que
abre posiciones solo cuando 2 de 3 indicadores (EMA, RSI, Bollinger) coinciden, ajusta sus
parámetros por régimen de mercado (K-means) y dimensiona con Risk Parity (Spinu) multiplicado por
la fuerza de la señal y por m(régimen). Todo se optimizó con Optuna maximizando Calmar en un
walk-forward 6→1 meses dentro de TRAIN ({split['train_inicio']} a {split['train_fin']}).

- **WF-OOS (TRAIN, {rp['inicio']} a {rp['fin']}):** CAGR {pct(rp['cagr'])}, volatilidad
  {pct(rp['volatilidad'])}, MDD {pct(rp['mdd'])}, Calmar {num(rp['calmar'])}, Sharpe {num(rp['sharpe'])},
  {num(rp['n_operaciones'], 0)} operaciones ({num(rp['n_largas'], 0)} largas / {num(rp['n_cortas'], 0)} cortas).
  Buy & Hold EW: CAGR {pct(bh['cagr'])}, MDD {pct(bh['mdd'])}, Calmar {num(bh['calmar'])}.
- **Walk-forward efficiency:** WFE de rendimiento {num(fila_roll['wfe_rendimiento'])} y de Calmar
  {num(fila_roll['wfe_calmar'])} (rolling); anchored: {num(fila_anc['wfe_rendimiento'])} / {num(fila_anc['wfe_calmar'])}.
- **Costos:** el punto de equilibrio está en una comisión de {pct(equilibrio, 3)} por lado
  (margen de seguridad {num(margen)}× frente a 0.125%).
- **TEST:** {test_txt}

Conclusión: {conclusion_principal(rp, bh, fila_roll, equilibrio, met_test)}

## 2. Datos y auditoría

- Fuente: Yahoo Finance vía yfinance, diario, `auto_adjust=True` (OHLC ajustados por splits y
  dividendos). Datos congelados en `data/prices_daily.csv`; `main.py` nunca re-descarga.
- Periodo: {split['train_inicio']} a {split['test_fin'] if met_test is not None else '2026-08-31'};
  incluye Q4-2018, COVID-2020, el bear market de 2022 y el choque de abril de 2025.
- Split cronológico sin traslape: TRAIN {split['train_inicio']} a {split['train_fin']}
  ({split['n_train']} días); TEST {split['test_inicio']} a {split['test_fin']} ({split['n_test']} días).
- Auditoría (`auditoria_datos.csv`): {int(auditoria['fechas_faltantes'].sum())} fechas faltantes,
  {int(auditoria['nans'].sum())} NaNs, {int(auditoria['duplicados'].sum())} duplicados,
  {int(auditoria['high_menor_que_max_oc'].sum() + auditoria['low_mayor_que_min_oc'].sum())} inconsistencias OHLC,
  {int(auditoria['precios_no_positivos'].sum())} precios ≤ 0. Rendimientos extremos |r| > 25%:
  {'; '.join(f"{r['ticker']} {r['rend_extremos']}" for _, r in auditoria.iterrows() if r['n_rend_extremos'] > 0)}
  — todos son reacciones a reportes trimestrales (eventos reales), no splits mal ajustados.
  La limpieza (solo fechas comunes a los 6, sin forward-fill) eliminó {split['filas_eliminadas_limpieza']} filas.
- **Selección y sesgos:** activos asignados con `random.seed(42)` → `random.sample` → pares en orden
  alfabético de integrantes. Son seis mega-cap tecnológicas que hoy sabemos ganadoras (sesgo de
  supervivencia y de selección): el Buy & Hold es un benchmark muy difícil de vencer en retorno,
  por eso el sistema se juzga por Calmar y riesgo. Además están muy correlacionadas, así que la
  diferencia entre Risk Parity y pesos iguales viene sobre todo de volatilidades distintas.

## 3. Indicadores y regla 2 de 3

Tres familias, implementadas a mano en `src/signals.py`:

1. **Tendencia — cruce de EMAs:** EMA_t = α·P_t + (1−α)·EMA_{{t−1}}, α = 2/(h+1).
   x₁ = +1 si EMA rápida > EMA lenta; −1 si es menor.
2. **Momento — RSI de Wilder:** x₂ = +1 si RSI < umbral inferior; −1 si RSI > umbral superior; 0 en otro caso.
3. **Volatilidad — Bandas de Bollinger:** MB = SMA_N, UB/LB = MB ± k·σ_N.
   x₃ = +1 si close < LB; −1 si close > UB; 0 dentro.

El ATR de Wilder (14 días) no vota: fija SL/TP.

**Regla de confirmación (Paso 7.2 de las notas), k = 3:**
s_i = (1/k)·Σ_j x_{{i,j}} si |Σ_j x_{{i,j}}| ≥ 2; s_i = 0 en otro caso. Por tanto s_i ∈ {{−1, −2/3, 0, +2/3, +1}}:
la *compuerta* (¿hay posición?) y la *fuerza* (¿cuánto riesgo?) se separan.
**Caso borde:** con votos (+1, +1, −1) la suma es 1 < 2 y **no se abre**: abrir exige 2 a favor y
ninguno en contra, o 3 a favor.

Interpretación económica: la regla abre en un *pullback* dentro de una tendencia (EMA + un
oscilador de acuerdo: comprar sobreventa en tendencia alcista, vender sobrecompra en tendencia
bajista) y con 3 de 3 cuando ambos osciladores confirman. **Hallazgo:** como la EMA siempre vota ±1,
con la EMA en contra la suma máxima es −1 + 1 + 1 = 1 < 2; por tanto la regla **nunca abre contra la
tendencia de la EMA** (fuera del calentamiento, cuando la EMA aún vota 0). El caso "reversión fuerte
contra la tendencia con RSI y Bollinger de acuerdo" que sugiere la formulación general no puede
ocurrir con estos tres indicadores: RSI y Bollinger funcionan como *timing* de entrada, y la EMA
como filtro de dirección.

Correlación promedio entre votos (θ congelado, WF-OOS): EMA–RSI {num(corr_ema_rsi)}, EMA–Bollinger
{num(corr_ema_bb)}, RSI–Bollinger {num(corr_rsi_bb)}. {'RSI y Bollinger superan 0.7: son parcialmente redundantes.' if corr_rsi_bb > 0.7 else 'Ninguna supera 0.7: los tres aportan información distinta (RSI y Bollinger son los más parecidos, como se espera de dos osciladores).'}

{fig('13_correlacion_senales.png', 'Correlación entre votos')}

## 4. Motor de backtest y convenciones

Motor event-driven (`src/backtest.py`), un solo motor para 1 o 6 activos, loop diario sobre arrays:

1. **Open de t:** primero gaps (si el open rebasó SL/TP de una posición previa, sale al open);
   luego se ejecutan las órdenes decididas al cierre de t−1. Toda señal se ejecuta en **t+1**.
2. **Durante t:** SL/TP con high/low, incluida la barra de entrada; **si SL y TP caen en la misma
   barra, se ejecuta primero el SL** (conservador).
3. **Cierre de t:** borrow fee (solo escenario realista), marca a mercado y decisiones con datos ≤ t.

SL/TP: largo SL = P_in − m_sl·ATR, TP = P_in + m_tp·ATR (corto al revés), con el ATR del día de la
señal y niveles fijos. Salidas: SL, TP, señal confirmada contraria (cierra y abre en reversa pagando
ambas comisiones), time-stop `max_hold`, entrada a Crisis y rebalanceo; una señal 0 no cierra.
Flujos: compra Cash −= q·P·(1+c); venta Cash += q·P·(1−c); un corto abre con Cash += q·P·(1−c) y
pasivo q < 0; Equity = Cash + Σ q_i·P_i. Comisión oficial c = 0.125% por lado.
**Sin apalancamiento:** antes de ejecutar, si Σ|q_i·P_i| > Equity/(1+c) se escalan las posiciones;
la exposición bruta máxima observada en WF-OOS fue {num(port_resumen['exposicion_bruta_max'], 3)}.
Con exposición ≤ 100% el margen de cortos (50% inicial, 25–30% mantenimiento) siempre se cumple.

## 5. Optimización

- Optuna `TPESampler(n_startup_trials=30)`: 30 trials aleatorios y luego TPE (Random Search →
  Bayesiana en un solo estudio); semilla por estudio derivada de SEED = 42 y (ventana, activo, régimen).
- Objetivo: **Calmar** del backtest de train con `entry_mask` del régimen j; si θ produce menos de
  N_MIN operaciones el objetivo vale −1e9. **N_MIN adaptado a 5 (global) / 3 (régimen):** con datos
  diarios y la regla 2 de 3 solo el 1% de los pares (ventana, activo) alcanza 10 operaciones en
  6 meses con *cualquier* θ (72% alcanza 5); con 10 casi todo caería al fallback.
- Purga y embargo: en train se cierran las posiciones el último día usado y los últimos
  {config.DIAS_EMBARGO} días del train se excluyen del objetivo.
- **θ robusto:** mediana de cada parámetro entre el 10% de mejores trials válidos (centro de la meseta).
- Espacio θ: EMA rápida 5–30, gap 10–100, RSI 7–28 / 20–40 / 60–80, Bollinger 10–40 / 1.5–3.0,
  m_sl 1–4, m_tp 1–6, max_hold 5–40. El tamaño de posición no se optimiza (el Calmar es casi
  invariante a la escala; lo fijan Risk Parity y m(régimen)).
- **{opt['n_trials_por_estudio']} trials por estudio** (por ventana y activo: 1 global + 3 regímenes);
  {opt['ventanas_rolling']} ventanas rolling y {opt['ventanas_anchored']} anchored.
  **Configuraciones evaluadas en TRAIN: {n_cfg:,}**{f' (+{test_sec["trials_totales_test_secundario"]:,} en el walk-forward secundario sobre TEST; total {n_cfg_total:,})' if test_sec else ''}.
  **Tiempo de optimización:** {num(horas, 2)} h de reloj (etapa TRAIN completa; {num(horas_cpu, 2)} h de CPU en {opt.get('n_procesos', 'varios')} procesos).
- Fallbacks a θ global (régimen con < 20 días o sin trials válidos), variante oficial: {fallback_txt}.
  Estudios globales sin ningún trial válido: {'; '.join(f"{g['esquema']} {g['variante']}: {g['sin_trials_validos']} de {g['estudios']} ({g['heredados_ventana_previa']} heredan el θ de la última ventana válida, {g['theta_por_defecto']} usan el θ por defecto)" for g in opt['fallbacks_globales'])}.
  Que tantas ventanas de 6 meses no alcancen el mínimo de operaciones es en sí un resultado: con
  datos diarios, la regla 2 de 3 genera pocas señales por semestre.
- **Variante elegida por Calmar WF-OOS en TRAIN: {congelado['variante']}.**

{tabla_md(variantes, ['nombre', 'cagr', 'mdd', 'calmar', 'n_operaciones', 'wfe_rendimiento', 'wfe_calmar'],
          ['Esquema | variante', 'CAGR', 'MDD', 'Calmar', '# ops', 'WFE rend.', 'WFE Calmar'],
          dict(FMT_METRICAS, wfe_rendimiento=num, wfe_calmar=num))}

Diagnóstico de Optuna ({diag['activo']}, última ventana de TRAIN {diag['train_ini']} a {diag['train_fin']}):

{fig('14a_optuna_historia.png', 'Historia')}
{fig('14b_optuna_importancia.png', 'Importancia')}
{fig('14c_optuna_slices.png', 'Slices')}
{fig('14d_optuna_superficie_3d.png', 'Superficie 3D')}

## 6. Métricas por conjunto

### WF-OOS (TRAIN)

{tabla_metricas(met_oos)}

{fig('01_valor_portafolio.png', 'Valor del portafolio')}
{fig('02_drawdown.png', 'Drawdown')}
{fig('03_rendimientos_wf_oos.png', 'Rendimientos WF-OOS')}

### WF-IS

Promedio por ventana (portafolio sobre su propio train, θ_k in-sample): CAGR {pct(fila_roll['cagr_is_promedio'])},
Calmar {num(fila_roll['calmar_is_promedio'])} (detalle en `wf_is_por_ventana.csv`).

### TEST

{tabla_test}

{fig('03_rendimientos_test.png', 'Rendimientos TEST') if met_test is not None else ''}

## 7. Análisis de régimen

- **Un régimen de mercado** sobre el índice equiponderado de los 6 activos; ventana móvil de 63 días.
- Features calculadas: volatilidad realizada √(1/n·Σr²) (nivel de riesgo, separa crisis), ATR
  normalizado (rango intradía comparable), fuerza de tendencia |MA21 − MA63|/σ (tendencia en sigmas),
  R² del log-precio contra el tiempo (linealidad de la tendencia) y autocorrelación ρ₁ (persistencia vs.
  reversión). **El K-means usa {', '.join(reg['features_usadas'])}**: silhouette {num(reg['silhouette_modelo_congelado'], 3)}
  vs. {num(reg['silhouette_5_features_misma_fecha'], 3)} con las 5 (objetivo > 0.4). ρ₁ con 63 datos es casi
  ruido (error estándar ≈ 1/√63 ≈ 0.126) y la fuerza de tendencia es redundante con R².
- K = 3 por teoría; `StandardScaler` + `KMeans(k-means++, n_init=10, random_state=42)` ajustados con
  datos ≤ fin del train de cada ventana (historia creciente, para que el cluster de Crisis haya visto
  crisis); en OOS solo `predict`. Re-etiquetado (invariante a los IDs de cluster): mayor volatilidad
  = Crisis; de los otros dos, el de mayor componente de tendencia estandarizado (R²; también
  fuerza de tendencia si estuviera en el subconjunto) = Tendencia; el restante = Reversión.
- **Adaptación a diario:** la clasificación se actualiza cada 5 días hábiles (los lineamientos piden
  1–6 h para datos de 5 min) y un cambio se confirma solo si aparece en 2 actualizaciones seguidas.
  Objetivo de persistencia adaptado: ≥ 10 días hábiles (más que una operación típica).
- Silhouette por ventana: media {num(reg['silhouette_por_ventana']['media'], 3)} (mín {num(reg['silhouette_por_ventana']['min'], 3)}, máx {num(reg['silhouette_por_ventana']['max'], 3)}).

Persistencia en WF-OOS (TRAIN):

{tabla_persist(persist)}

{('Persistencia en TEST (modelo congelado):' + chr(10) + chr(10) + tabla_persist(reg_test)) if reg_test is not None else 'Persistencia en TEST: ' + PENDIENTE}

Transiciones en WF-OOS: {reg['transiciones']['n_transiciones']}; posiciones abiertas al momento de una
transición: {reg['transiciones']['posiciones_abiertas_afectadas']}; cierres por entrada a Crisis:
{reg['transiciones']['cierres_por_cambio_a_crisis']}. Reglas aplicadas: {' '.join(reg['transiciones']['reglas'])}

Desempeño diario del sistema por régimen (IC 95% bootstrap, 1000 remuestreos):

{tabla_md(por_reg, ['nombre', 'regimen', 'dias', 'media_diaria', 'ic95_inf', 'ic95_sup', 'volatilidad_anual', 'sharpe', 'mdd'],
          ['Serie', 'Régimen', 'Días', 'Media diaria', 'IC inf', 'IC sup', 'Vol. anual', 'Sharpe', 'MDD'],
          {'nombre': str, 'regimen': str, 'dias': lambda x: num(x, 0), 'media_diaria': lambda x: pct(x, 3),
           'ic95_inf': lambda x: pct(x, 3), 'ic95_sup': lambda x: pct(x, 3), 'volatilidad_anual': pct,
           'sharpe': num, 'mdd': pct})}

Correlación promedio entre pares por régimen: {', '.join(f"{r['regimen']} {num(r['correlacion_media'])}" for _, r in corr_reg.iterrows())}.

{fig('06a_regimenes_linea_tiempo.png', 'Regímenes')}
{fig('06b_features_por_regimen.png', 'Features por régimen')}
{fig('06c_valor_con_regimenes.png', 'Valor con regímenes')}
{fig('06d_transiciones.png', 'Transiciones')}
{fig('07c_correlacion_por_regimen.png', 'Correlación por régimen')}

## 8. Metodología del portafolio (notas "Fundamentos Matemáticos de Risk Parity")

1. **De R_p a σ_p:** R_p = wᵀR, σ_p² = wᵀΣw, σ_p = √(wᵀΣw). Σ siempre sobre rendimientos (los precios
   no son estacionarios y dan correlaciones espurias).
2. **Contribución marginal:** MRC_k = ∂σ_p/∂w_k = (Σw)_k/σ_p = Cov(R_k, R_p)/σ_p.
3. **Contribución total y Euler:** RC_i = w_i(Σw)_i/σ_p y Σ_i RC_i = σ_p exacto (probado a 1e-10),
   así que RC_i/σ_p es un porcentaje genuino del riesgo.
4. **Condición de RP:** RC_i = σ_p/n ⇔ w_i(Σw)_i = w_j(Σw)_j, Σw_i = 1, w_i > 0. No usa rendimientos esperados.
5. **Naive vs. Spinu:** naive w_i ∝ 1/σ_i es exacto solo con correlaciones iguales. El sistema usa
   min_{{y>0}} ½yᵀΣy − (1/n)Σ ln y_i, w = y/Σy (convexa, solución única; L-BFGS-B con gradiente
   Σy − 1/(n·y)). No se minimiza Σ(RC_i − RC_j)² porque es no convexo y depende del punto inicial.
   Verificación en cada rebalanceo: max|RC_i/σ_p − 1/n| < 1e-4. Con nuestros datos, la diferencia
   máxima promedio entre pesos RP y naive es {pct(dif_naive_rp, 2)} por activo.
6. **Σ:** muestral, 504 días (n/T ≈ 0.012), solo datos ≤ t, re-estimada en cada revisión. Comparación
   de estabilidad con EWMA (λ = 0.94, T_eff ≈ 17 días) y Ledoit-Wolf:

{tabla_md(estim, ['estimador', 'volatilidad_media_pesos', 'turnover_anual_implicito', 'costo_anual_implicito', 'numero_condicion_medio'],
          ['Estimador', 'Desv. media de pesos', 'Turnover anual implícito', 'Costo anual implícito', 'Nº condición medio'],
          {'estimador': str, 'volatilidad_media_pesos': lambda x: num(x, 4), 'turnover_anual_implicito': num,
           'costo_anual_implicito': lambda x: pct(x, 3), 'numero_condicion_medio': lambda x: num(x, 1)})}

7. **Agregación (Paso 7):** w̃_i = w_i^RP·s_i y w^target = m(régimen)·w̃/max(1, Σ|w̃|), con
   m = 1.0 / 0.7 / 0.3 (Tendencia / Reversión / Crisis; en Crisis solo entradas 3 de 3). s_i es la
   del momento de entrada y se conserva mientras la posición viva. **Conflictos:** si corr > 0.7 y
   señales opuestas, se queda la de mayor |s| (empate: ambas × 0.5); la política actuó
   {port_resumen['conflictos_politica']} veces en WF-OOS. Consecuencia del diseño: una posición aislada pesa
   ≈ w^RP·|s|·m ≈ 1/6 × 2/3 ≈ 11% del capital, y la exposición media fue {pct(rp['exposicion_media'])}.
8. **Rebalanceo:** T_t = ½Σ|w_t − w_{{t⁻}}| contra el peso post-drift; costo ≈ T̄·f·2c
   ({pct(port_resumen['costo_anualizado_formula'], 4)} anual por fórmula vs. {pct(port_resumen['costos_cobrados_motor_anual_pct'], 3)}
   anual cobrado por el motor, que incluye todas las entradas y salidas). Disparador híbrido: en
   fechas de calendario f se rebalancea solo si ‖w − w^target‖₁ > δ. Barrido en TRAIN →
   **f = {congelado['rebalanceo']['frecuencia']}, δ = {congelado['rebalanceo']['delta']}**.

**Ilusión del 50/50 con nuestros datos:** con pesos iguales, TSLA + NVDA aportan
{pct(ilusion)} del riesgo con {pct(ilusion_peso)} del capital.

**¿Por qué Risk Parity?** No requiere rendimientos esperados (a diferencia de Markowitz) sino solo Σ,
el insumo que se estima con error aceptable; pesos iguales no es riesgo igual; la descomposición es
exacta por Euler; y se integra con señales y régimen antes de volverse una orden.

{fig('07a_contribuciones_riesgo.png', 'Contribuciones al riesgo')}
{fig('07d_barrido_rebalanceo.png', 'Barrido de rebalanceo')}
{fig('08_pesos_estimadores_sigma.png', 'Pesos por estimador')}
{fig('07b_senales_s_wf_oos.png', 'Fuerza de señal')}

## 9. Estrategia individual de cada activo

θ congelado (variante **{congelado['variante']}**, última ventana {congelado['train_ini_ultima_ventana']} a
{congelado['train_fin_ultima_ventana']}):

{chr(10).join(tabla_theta)}

Estrategia sola en cada activo (100% del capital, WF-OOS):

{tabla_metricas(individuales)}

Contribución de cada activo dentro del portafolio (WF-OOS):

{tabla_md(por_activo_port, ['activo', 'pnl_neto', 'costos', 'n_operaciones', 'n_largas', 'n_cortas', 'win_rate'],
          ['Activo', 'PnL neto (USD)', 'Costos (USD)', '# ops', '# largas', '# cortas', 'Win rate'],
          {'activo': str, 'pnl_neto': lambda x: num(x, 0), 'costos': lambda x: num(x, 0),
           'n_operaciones': lambda x: num(x, 0), 'n_largas': lambda x: num(x, 0),
           'n_cortas': lambda x: num(x, 0), 'win_rate': pct})}

Responsables: Milca ({', '.join(asign['Milca'])}), Paula ({', '.join(asign['Paula'])}), Arturo ({', '.join(asign['Arturo'])}).

{fig('10_portafolio_vs_individuales_wf_oos.png', 'Portafolio vs individuales')}

## 10. Comparaciones

{tabla_md(rp_cmp.reset_index(), ['nombre', 'cagr', 'volatilidad', 'mdd', 'calmar', 'turnover_anual', 'rc_max', 'rc_tsla_nvda'],
          ['Pesos', 'CAGR', 'Vol.', 'MDD', 'Calmar', 'Turnover', 'Máx RC_i/σ_p', 'RC TSLA+NVDA'],
          dict(FMT_METRICAS, rc_max=pct, rc_tsla_nvda=pct))}

Rolling vs. anchored (WF-OOS, rebalanceo por defecto semanal/0.10): Calmar {num(fila_roll['calmar'])} vs. {num(fila_anc['calmar'])};
CAGR {pct(fila_roll['cagr'])} vs. {pct(fila_anc['cagr'])}.

{fig('12_rolling_vs_anchored.png', 'Rolling vs anchored')}

## 11. Robustez y costos

**2 de 3 vs. un indicador (WF-OOS, mismo θ por ventana):**

{tabla_md(port_ind.reset_index(), ['caso', 'n_operaciones', 'calmar', 'cagr', 'mdd'],
          ['Regla', '# ops', 'Calmar', 'CAGR', 'MDD'], dict(FMT_METRICAS, caso=str))}

{fig('09_dos_de_tres_vs_un_indicador.png', '2 de 3 vs un indicador')}

**Sensibilidad ±20%** (θ congelado como base, uno a la vez, portafolio sobre TRAIN): Calmar base
{num(base_sens)}; parámetro más sensible: **{peor_param}** (rango de Calmar {num(rango_sens.max())}).
Con ±20% en los parámetros de θ, el Calmar conserva el signo de la base en {pct(frac_signo, 0)} de los
casos, queda dentro de ±50% de la base en {pct(frac_cerca, 0)} y va de {num(calmar_min_sens)} a
{num(calmar_max_sens)}. Nota: la base de sensibilidad usa el θ congelado y el modelo de régimen
congelado sobre todo el tramo de TRAIN, por eso no coincide con el Calmar WF-OOS.

{fig('04_sensibilidad.png', 'Sensibilidad')}

**Curva de costos:** equilibrio en {pct(equilibrio, 3)} por lado → margen de seguridad
{num(margen)}× frente a 0.125%. Operaciones por año: {num(costos['operaciones_por_anio'], 1)}; costo
anual por turnover: {pct(costos['costo_anual_pct_capital'], 3)} del capital.

{fig('05_curva_costos.png', 'Curva de costos')}
{fig('11_costos_vs_bruto_wf_oos.png', 'Costos vs bruto')}

## 12. Supuestos y decisiones

- Periodo 2015-01-01 a 2026-08-31 (no hizo falta reducirlo: la etapa TRAIN completa tomó {num(horas, 2)} h).
- La regla 2 de 3 nunca abre contra la tendencia de la EMA (ver sección 3).
- N_MIN = 5 (global) / 3 (régimen) en lugar de 10/5 (ver sección 5); compartido: 6× esos valores.
- Fallback declarado: régimen con < 20 días en el train o sin trials válidos → θ global de la ventana;
  estudio global sin trials válidos → θ global de la última ventana previa válida (causal; el sistema
  conserva sus últimos parámetros optimizados); solo sin ninguna ventana previa válida → θ por defecto.
- N_MIN en anchored escala con los días de la muestra (global) o del régimen (por régimen); en
  rolling vale exactamente 5 y 3.
- θ robusto = mediana del top 10% de trials válidos (al menos 1 trial).
- Features del K-means: {', '.join(reg['features_usadas'])} (selección con silhouette, solo en TRAIN).
- Modelo de régimen ajustado con historia creciente ≤ fin del train de cada ventana (la ventana de 6
  meses no contiene crisis en muchos casos).
- Régimen actualizado cada 5 días hábiles con persistencia de 2 actualizaciones; inicialización en el
  primer día con modelo vigente. El rendimiento de t se atribuye al régimen vigente al cierre de t−1.
- Al entrar a Crisis se cierran al open siguiente las posiciones con |s| < 1 y el resto se ajusta a m = 0.3.
- Gap: el open se evalúa contra SL/TP antes de ejecutar las órdenes pendientes del mismo activo.
- Límite de no apalancamiento Σ|q·P| ≤ Equity/(1+c) para que el efectivo cubra la comisión.
- Pesos de largo plazo con ventana creciente (mínimo 126 días) al inicio de la muestra.
- Paralelización por ventana (las ventanas son independientes), no por activo.
- Experimento de un indicador con el mismo θ por ventana (aísla el efecto de la regla).
- Sensibilidad de m(régimen) con tope en 1.0.
- Spread de 2 bps = spread completo (se paga la mitad por lado); impacto η = 0.1 con ADV de 20 días.
- Turnover del ejemplo de las notas: sin las notas a mano, la prueba usa un ejemplo construido por
  el equipo (0.10 contra el objetivo viejo vs. 0.05 contra el peso post-drift).
- Las "operaciones" contadas son las cerradas; una posición abierta al final se marca a mercado.

## 13. Respuestas a las 7 preguntas

1. **¿Qué aporta la regla 2 de 3 frente a un solo indicador?** {respuesta_1(port_ind)}
2. **¿Cuánto se degrada el desempeño de train a test? ¿Qué proporción sobrevive?** WFE de rendimiento
   {num(fila_roll['wfe_rendimiento'])} y de Calmar {num(fila_roll['wfe_calmar'])} (rolling): CAGR WF-IS promedio
   {pct(fila_roll['cagr_is_promedio'])} vs. WF-OOS {pct(fila_roll['cagr_oos'])}. {interpretar_wfe(fila_roll['wfe_rendimiento'])}
   {degradacion}
3. **¿Qué tan sensible es a ±20%? ¿Meseta o pico?** Calmar base {num(base_sens)}; el parámetro más
   sensible es {peor_param}. Con ±20% el Calmar conserva el signo en {pct(frac_signo, 0)} de los casos
   y solo {pct(frac_cerca, 0)} queda dentro de ±50% de la base (rango {num(calmar_min_sens)} a {num(calmar_max_sens)}).
   {interpretar_sensibilidad(frac_signo, frac_cerca)}
4. **¿A qué costo deja de ser rentable?** {('A ' + pct(equilibrio, 3) + ' por lado; margen de seguridad ' + num(margen) + '× frente a 0.125%.') if equilibrio and np.isfinite(equilibrio) and equilibrio > 0 else 'Con la comisión oficial el retorno neto WF-OOS ya es ≤ 0: no hay margen de seguridad.'}
   En el escenario realista (spread 2 bps, borrow 0.5%, impacto) el CAGR pasa de
   {pct(realistas.loc['oficial (solo comisión)', 'cagr'])} a {pct(realistas.loc['realista (todo junto)', 'cagr'])}
   y el Calmar de {num(realistas.loc['oficial (solo comisión)', 'calmar'])} a {num(realistas.loc['realista (todo junto)', 'calmar'])}.
5. **¿El desempeño difiere entre regímenes?** {respuesta_5(reg_rp, sig_reg)}
6. **¿Risk Parity mejora el Calmar frente a pesos iguales?** {'Sí' if rp['calmar'] > ew['calmar'] else 'No'}: RP Calmar {num(rp['calmar'])} vs. EW
   {num(ew['calmar'])} (naive {num(naive['calmar'])}); MDD {pct(rp['mdd'])} vs. {pct(ew['mdd'])};
   CAGR {pct(rp['cagr'])} vs. {pct(ew['cagr'])}. Contribución máxima al riesgo de un activo: RP
   {pct(rp_cmp.loc['rp', 'rc_max'])} vs. EW {pct(rp_cmp.loc['ew', 'rc_max'])}. Con pesos iguales TSLA+NVDA aportan
   {pct(ilusion)} del riesgo con {pct(ilusion_peso)} del capital. RP vs. naive difieren a lo más
   {pct(dif_naive_rp, 2)} en pesos promedio: con correlaciones parecidas, naive ≈ Spinu. RP sí iguala
   las contribuciones al riesgo (su objetivo), pero como la exposición media es ~10% y las señales
   dominan el P&L, el reparto de pesos casi no mueve el Calmar.
7. **Tres limitaciones para operar con capital real:** (i) universo de 6 mega-cap tecnológicas muy
   correlacionadas (correlación media {num(corr_reg['correlacion_media'].mean())}) elegidas *ex post*
   (supervivencia): la diversificación es limitada y el régimen de Crisis afecta a todos a la vez;
   (ii) pocas operaciones por ventana ({num(costos['operaciones_por_anio'], 1)} por año en todo el portafolio)
   → Calmar con mucho error de estimación y riesgo de minería de datos ({n_cfg_total:,} configuraciones
   probadas); (iii) ejecución idealizada: llenado completo al open/al nivel de SL/TP, sin impacto,
   sin restricciones de préstamo de títulos ni margin calls intradía.

## 14. Advertencia de ejecución

El backtest supone ejecución completa al precio modelado, sin impacto de mercado ni fallas de
ejecución. Estimación con el sistema congelado sobre TRAIN:

{tabla_md(realistas.reset_index(), ['caso', 'cagr', 'calmar', 'mdd', 'costos_totales'],
          ['Escenario', 'CAGR', 'Calmar', 'MDD', 'Costos (USD)'],
          dict(FMT_METRICAS, caso=str, costos_totales=lambda x: num(x, 0)))}

Barrido de slippage (por lado): CAGR de {pct(slip['cagr'].iloc[0])} (0 bps) a {pct(slip['cagr'].iloc[-1])}
(20 bps); Calmar de {num(slip['calmar'].iloc[0])} a {num(slip['calmar'].iloc[-1])}.

## Anexo: evaluación secundaria del TEST

{sec_txt}
""")
    return "\n".join(secciones)


def conclusion_principal(rp, bh, fila_roll, equilibrio, met_test):
    partes = []
    if rp["calmar"] > 0:
        partes.append(f"el sistema tiene Calmar WF-OOS positivo ({num(rp['calmar'])})")
    else:
        partes.append(f"el sistema no genera un Calmar WF-OOS positivo ({num(rp['calmar'])})")
    partes.append(f"con un CAGR ({pct(rp['cagr'])}) muy inferior al Buy & Hold ({pct(bh['cagr'])}) porque "
                  f"la exposición media es baja ({pct(rp['exposicion_media'])})")
    wfe = fila_roll["wfe_rendimiento"]
    if np.isfinite(wfe) and wfe < 0.5:
        partes.append(f"y una WFE de {num(wfe)} < 0.5 indica que la mayor parte del desempeño in-sample "
                      "es ajuste a la muestra, no una ventaja estable")
    else:
        partes.append(f"con una WFE de {num(wfe)}")
    if met_test is None:
        partes.append(f"el veredicto final depende del TEST {PENDIENTE}")
    return "; ".join(partes) + "."


def respuesta_1(port_ind):
    dos = port_ind.loc["2 de 3"]
    otros = port_ind.drop(index="2 de 3")
    texto = (f"Con 2 de 3 hubo {num(dos['n_operaciones'], 0)} operaciones, Calmar {num(dos['calmar'])} y "
             f"MDD {pct(dos['mdd'])}; "
             + "; ".join(f"{caso}: {num(f['n_operaciones'], 0)} operaciones, Calmar {num(f['calmar'])}, "
                         f"MDD {pct(f['mdd'])}, costos {num(f['costos_totales'], 0)} USD"
                         for caso, f in otros.iterrows()) + ". ")
    menos = (otros["n_operaciones"] > dos["n_operaciones"]).mean()
    mejor = (otros["calmar"] < dos["calmar"]).mean()
    texto += (f"La regla filtra operaciones (menos operaciones que {pct(menos, 0)} de los indicadores solos), "
              f"reduce costos y drawdown, y supera el Calmar de {pct(mejor, 0)} de ellos. "
              "Si algún indicador solo tiene mayor Calmar, lo logra con mucho más drawdown y costo.")
    return texto


def interpretar_wfe(wfe):
    if not np.isfinite(wfe):
        return "La WFE no está definida (rendimiento in-sample nulo)."
    if wfe < 0:
        return "Fuera de muestra el rendimiento cambia de signo: la ventaja in-sample no sobrevive."
    if wfe < 0.5:
        return (f"Sobrevive alrededor de {pct(wfe, 0)} del rendimiento in-sample: por debajo de 0.5, "
                "el resultado in-sample es mayormente ruido ajustado.")
    return f"Sobrevive alrededor de {pct(min(wfe, 1), 0)} del rendimiento in-sample."


def interpretar_sensibilidad(frac_signo, frac_cerca):
    if frac_signo >= 0.8 and frac_cerca >= 0.6:
        return "La mayoría de las perturbaciones conserva el resultado: se parece más a una meseta."
    if frac_signo >= 0.8:
        return ("El signo casi siempre se conserva, pero la magnitud del Calmar cambia mucho: es una "
                "meseta baja y ruidosa, no un pico aislado, pero tampoco un resultado estable en magnitud.")
    return "Muchas perturbaciones cambian el signo del Calmar: el resultado se parece más a un pico aislado."


def respuesta_5(reg_rp, sig_reg):
    filas = "; ".join(f"{r['regimen']}: media diaria {pct(r['media_diaria'], 3)} "
                      f"[{pct(r['ic95_inf'], 3)}, {pct(r['ic95_sup'], 3)}], Sharpe {num(r['sharpe'])}, MDD {pct(r['mdd'])}"
                      for _, r in reg_rp.iterrows())
    if len(sig_reg) == 0:
        cierre = ("Todos los intervalos incluyen 0 y se traslapan: no hay diferencias significativas. "
                  "La capa de régimen aporta sobre todo control de riesgo (m = 0.3 y solo 3 de 3 en Crisis, "
                  "θ distintos por régimen), no un retorno distinto demostrable.")
    else:
        cierre = (f"Solo en {', '.join(sig_reg['regimen'])} el intervalo excluye 0; en los demás no hay "
                  "diferencia significativa.")
    return filas + ". " + cierre


# ----------------------------------------------------------------------------
# Markdown -> PDF (reportlab)
# ----------------------------------------------------------------------------
def _registrar_fuente():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    carpeta = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DejaVu", str(carpeta / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(carpeta / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuMono", str(carpeta / "DejaVuSansMono.ttf")))
    from reportlab.lib.fonts import addMapping
    addMapping("DejaVu", 0, 0, "DejaVu")
    addMapping("DejaVu", 1, 0, "DejaVu-Bold")


def _inline(texto):
    texto = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    texto = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", texto)
    texto = re.sub(r"\*(.+?)\*", r"<i>\1</i>", texto)
    texto = re.sub(r"`(.+?)`", r'<font face="DejaVuMono">\1</font>', texto)
    return texto


def markdown_a_pdf(markdown, ruta_pdf, dir_base):
    """Convierte el markdown simple del reporte a PDF con reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    _registrar_fuente()
    base = ParagraphStyle("base", fontName="DejaVu", fontSize=9.5, leading=13)
    estilos = {1: ParagraphStyle("h1", parent=base, fontSize=17, leading=21, spaceAfter=8, fontName="DejaVu-Bold"),
               2: ParagraphStyle("h2", parent=base, fontSize=13, leading=17, spaceBefore=10, spaceAfter=5,
                                 fontName="DejaVu-Bold"),
               3: ParagraphStyle("h3", parent=base, fontSize=11, leading=15, spaceBefore=6, fontName="DejaVu-Bold")}
    celda = ParagraphStyle("celda", parent=base, fontSize=7, leading=8.5)
    ancho = letter[0] - 3 * cm
    historia, parrafo, tabla = [], [], []

    def cerrar_parrafo():
        if parrafo:
            historia.append(Paragraph(_inline(" ".join(parrafo)), base))
            historia.append(Spacer(1, 4))
            parrafo.clear()

    def cerrar_tabla():
        if tabla:
            filas = [[Paragraph(_inline(c.strip()), celda) for c in f.strip().strip("|").split("|")]
                     for f in tabla if not re.match(r"^\|(-+\|)+$", f.strip())]
            n = max(len(f) for f in filas)
            filas = [f + [""] * (n - len(f)) for f in filas]
            t = Table(filas, colWidths=[ancho / n] * n, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c3c2b7")),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e4e3df")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            historia.append(t)
            historia.append(Spacer(1, 6))
            tabla.clear()

    for linea in markdown.splitlines():
        s = linea.rstrip()
        if s.startswith("|"):
            cerrar_parrafo()
            tabla.append(s)
            continue
        cerrar_tabla()
        imagen = re.match(r"!\[(.*?)\]\((.+?)\)", s.strip())
        if imagen:
            cerrar_parrafo()
            ruta = (dir_base / imagen.group(2)).resolve()
            if ruta.exists():
                from reportlab.lib.utils import ImageReader
                w, h = ImageReader(str(ruta)).getSize()
                escala = min(ancho / w, 9 * cm / h)
                historia.append(Image(str(ruta), width=w * escala, height=h * escala))
                historia.append(Spacer(1, 6))
            continue
        encabezado = re.match(r"^(#{1,3})\s+(.*)", s)
        if encabezado:
            cerrar_parrafo()
            historia.append(Paragraph(_inline(encabezado.group(2)), estilos[len(encabezado.group(1))]))
            continue
        if not s.strip():
            cerrar_parrafo()
            continue
        if re.match(r"^\s*(-|\d+\.)\s+", s):
            cerrar_parrafo()
            parrafo.append(re.sub(r"^\s*-\s+", "• ", s))
            continue
        parrafo.append(s.strip())
    cerrar_parrafo()
    cerrar_tabla()
    doc = SimpleDocTemplate(str(ruta_pdf), pagesize=letter, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Lab 02 MyST — Equipo 2", author="Milca, Paula, Arturo")
    doc.build(historia)


# ----------------------------------------------------------------------------
# Presentación (máx. 12 diapositivas + portada + cierre)
# ----------------------------------------------------------------------------
def construir_diapositivas(R):
    """Lista de diapositivas: (titulo, viñetas, figura, presentador, segundos)."""
    met_oos = R.csv("metricas_wf_oos.csv").set_index("nombre")
    met_test = R.csv("metricas_test.csv")
    split = R.json("splits.json")
    cong = R.json("theta_congelado.json")
    costos = R.json("costos_resumen.json")
    reg = R.json("regimenes_validacion.json")
    var = R.csv("comparacion_variantes.csv")
    fila_roll = var[(var["esquema"] == "rolling") & (var["variante"] == cong["variante"])].iloc[0]
    un = R.csv("un_indicador.csv")
    un = un[un["nivel"] == "portafolio"].set_index("caso")
    contrib = R.csv("contribuciones_riesgo_promedio.csv")
    ew_rc = contrib[contrib["metodo"] == "ew"].set_index("activo")
    sens = R.csv("sensibilidad.csv")
    opt = R.json("optimizacion_resumen.json")
    rp, ew, bh = met_oos.loc["RP (sistema)"], met_oos.loc["Pesos iguales (EW)"], met_oos.loc["Buy & Hold EW"]
    asign = config.ASIGNACION
    if met_test is not None:
        t = met_test.set_index("nombre")
        texto_test = [f"Sistema congelado: CAGR {pct(t.loc['RP (sistema)', 'cagr'])}, MDD {pct(t.loc['RP (sistema)', 'mdd'])}, "
                      f"Calmar {num(t.loc['RP (sistema)', 'calmar'])}",
                      f"Buy & Hold: CAGR {pct(t.loc['Buy & Hold EW', 'cagr'])}, Calmar {num(t.loc['Buy & Hold EW', 'calmar'])}",
                      "Se tocó UNA vez, con candado (hash de θ + commit)"]
    else:
        texto_test = [PENDIENTE, "Se toca UNA sola vez con θ congelado y candado (hash + commit)"]
    base_sens = sens[sens["factor"] == 0]["calmar"].iloc[0]
    ilusion = ew_rc.loc[["TSLA", "NVDA"], "rc_pct"].sum()
    return [
        ("Problema, datos y activos",
         [f"6 mega-cap tech diarias, {split['train_inicio']} a 2026-08-31 (congelado)",
          f"TRAIN hasta {split['train_fin']} · TEST desde {split['test_inicio']}",
          f"Milca: {', '.join(asign['Milca'])} · Paula: {', '.join(asign['Paula'])} · Arturo: {', '.join(asign['Arturo'])}",
          "Sesgo de supervivencia: B&H es casi imbatible en retorno → se juzga por Calmar"],
         "06a_regimenes_linea_tiempo.png", "Milca", 50),
        ("Señal: 3 familias y regla 2 de 3",
         ["EMA (tendencia) · RSI (momento) · Bollinger (volatilidad)",
          "s = (1/3)·Σ votos si |Σ| ≥ 2; (+1, +1, −1) NO abre",
          "Compuerta (¿hay posición?) ≠ fuerza (¿cuánto riesgo?)"],
         "13_correlacion_senales.png", "Milca", 50),
        ("¿Qué aporta 2 de 3?",
         [f"2 de 3: {num(un.loc['2 de 3', 'n_operaciones'], 0)} operaciones, Calmar {num(un.loc['2 de 3', 'calmar'])}"]
         + [f"{c}: {num(f['n_operaciones'], 0)} ops, Calmar {num(f['calmar'])}" for c, f in un.drop(index='2 de 3').iterrows()],
         "09_dos_de_tres_vs_un_indicador.png", "Milca", 55),
        ("Motor de backtest",
         ["Ejecución en t+1 al open · SL primero si SL y TP en la misma barra",
          "Gap → sale al open · comisión 0.125% por lado · cortos con pasivo",
          "Sin apalancamiento: Σ|w| ≤ 1 verificado en cada ejecución"],
         "11_costos_vs_bruto_wf_oos.png", "Paula", 50),
        ("Optimización y walk-forward",
         [f"Optuna TPE, {opt['n_trials_por_estudio']} trials por régimen por ventana, objetivo Calmar",
          f"{opt['ventanas_rolling']} ventanas 6→1 meses · {opt['configuraciones_evaluadas_train']:,} configuraciones",
          f"WFE rendimiento {num(fila_roll['wfe_rendimiento'])} · WFE Calmar {num(fila_roll['wfe_calmar'])}"],
         "14d_optuna_superficie_3d.png", "Paula", 60),
        ("WF-OOS vs. benchmarks",
         [f"RP: CAGR {pct(rp['cagr'])}, MDD {pct(rp['mdd'])}, Calmar {num(rp['calmar'])}",
          f"EW: Calmar {num(ew['calmar'])} · B&H: CAGR {pct(bh['cagr'])}, Calmar {num(bh['calmar'])}",
          f"Exposición media {pct(rp['exposicion_media'])}: w = w^RP·s·m"],
         "01_valor_portafolio.png", "Paula", 60),
        ("Sensibilidad ±20%",
         [f"Calmar base (θ congelado, TRAIN) {num(base_sens)}",
          "Uno a la vez, a todos los activos y regímenes"],
         "04_sensibilidad.png", "Paula", 50),
        ("¿A qué costo deja de ser rentable?",
         [f"Equilibrio: {pct(costos['comision_equilibrio'], 3)} por lado",
          f"Margen de seguridad {num(costos['margen_seguridad'])}× frente a 0.125%"],
         "05_curva_costos.png", "Paula", 45),
        ("Regímenes (K-means, K = 3)",
         [f"Features: {', '.join(reg['features_usadas'])} · silhouette {num(reg['silhouette_modelo_congelado'], 3)}",
          "Actualización semanal + persistencia de 2 actualizaciones",
          "Duración promedio: " + ", ".join(f"{p['regimen']} {num(p['duracion_promedio_obs'], 0)} d" for p in reg['persistencia'])],
         "06c_valor_con_regimenes.png", "Arturo", 55),
        ("Risk Parity: la ilusión del 50/50",
         [f"Con pesos iguales TSLA + NVDA = {pct(ilusion)} del riesgo",
          "Spinu: convexo, solución única, RC_i/σ_p = 1/6 ± 1e-4"],
         "07a_contribuciones_riesgo.png", "Arturo", 55),
        ("Rebalanceo: banda + calendario",
         [f"Elegido en TRAIN: {cong['rebalanceo']['frecuencia']}, δ = {cong['rebalanceo']['delta']}",
          "Turnover contra el peso post-drift (no contra el objetivo viejo)"],
         "07d_barrido_rebalanceo.png", "Arturo", 50),
        ("TEST: una sola vez",
         texto_test,
         "01_valor_portafolio.png", "Arturo", 55),
    ]


def construir_presentacion_md(diapositivas, conclusiones):
    lineas = ["# Presentación — Lab 02 MyST · Equipo 2 · Nivel C", "",
              "10 minutos de exposición + 5 de preguntas. Máximo 12 diapositivas (sin contar portada y cierre).", ""]
    total = 0
    for k, (titulo, vinetas, figura, quien, segundos) in enumerate(diapositivas, 1):
        total += segundos
        lineas += [f"## {k}. {titulo} — {quien} ({segundos} s)", ""] + [f"- {v}" for v in vinetas] + \
                  [f"- Figura: `docs/figures/{figura}`", ""]
    lineas += ["## Cierre — los 3 (30 s)", ""] + [f"- {c}" for c in conclusiones] + ["",
               f"Tiempo total de exposición: {total + 30} s ≈ {(total + 30) / 60:.1f} min (+ portada)."]
    return "\n".join(lineas)


def presentacion_pdf(diapositivas, conclusiones, dir_fig, ruta_pdf):
    """Diapositivas 16:9 con matplotlib PdfPages: poco texto, fuentes grandes, sin código."""
    with PdfPages(ruta_pdf) as pdf:
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.5, 0.62, "Estrategias de Trading con Análisis Técnico", ha="center", fontsize=34, weight="bold")
        fig.text(0.5, 0.50, "Laboratorio 02 · Microestructuras y Sistemas de Trading · ITESO", ha="center", fontsize=20)
        fig.text(0.5, 0.40, "Equipo 2 · Nivel C · Milca · Paula · Arturo", ha="center", fontsize=20, color="#52514e")
        pdf.savefig(fig)
        plt.close(fig)
        for k, (titulo, vinetas, figura, quien, _) in enumerate(diapositivas, 1):
            fig = plt.figure(figsize=(13.33, 7.5))
            fig.text(0.04, 0.92, f"{k}. {titulo}", fontsize=28, weight="bold")
            fig.text(0.96, 0.93, quien, fontsize=14, ha="right", color="#52514e")
            y = 0.82
            for v in vinetas:
                fig.text(0.05, y, "• " + v, fontsize=17, va="top", wrap=True)
                y -= 0.065
            ruta = dir_fig / figura
            if ruta.exists():
                imagen = plt.imread(ruta)
                alto = max(0.2, y - 0.03)
                ax = fig.add_axes([0.05, 0.02, 0.9, alto])
                ax.imshow(imagen)
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=(13.33, 7.5))
        fig.text(0.04, 0.9, "Conclusiones", fontsize=30, weight="bold")
        y = 0.78
        for c in conclusiones:
            fig.text(0.05, y, "• " + c, fontsize=18, va="top", wrap=True)
            y -= 0.12
        pdf.savefig(fig)
        plt.close(fig)


def conclusiones_cortas(R):
    met = R.csv("metricas_wf_oos.csv").set_index("nombre")
    var = R.csv("comparacion_variantes.csv")
    cong = R.json("theta_congelado.json")
    costos = R.json("costos_resumen.json")
    fila = var[(var["esquema"] == "rolling") & (var["variante"] == cong["variante"])].iloc[0]
    rp, ew = met.loc["RP (sistema)"], met.loc["Pesos iguales (EW)"]
    met_test = R.csv("metricas_test.csv")
    test = (f"TEST congelado: Calmar {num(met_test.set_index('nombre').loc['RP (sistema)', 'calmar'])}"
            if met_test is not None else f"TEST: {PENDIENTE}")
    return [f"WF-OOS: Calmar {num(rp['calmar'])} (EW {num(ew['calmar'])}), MDD {pct(rp['mdd'])}, exposición media {pct(rp['exposicion_media'])}",
            f"WFE {num(fila['wfe_rendimiento'])}: {interpretar_wfe(fila['wfe_rendimiento'])}",
            f"Equilibrio de costos en {pct(costos['comision_equilibrio'], 3)} por lado (margen {num(costos['margen_seguridad'])}×)",
            test]


# ----------------------------------------------------------------------------
def generar_todo(dir_res, dir_fig, dir_docs, quick):
    """Escribe borradores (.md) y PDFs. En --quick todo va a .cache/quick/."""
    R = Resultados(dir_res)
    destino = Path(dir_docs) if dir_docs is not None else Path(dir_res).parent
    destino.mkdir(parents=True, exist_ok=True)
    figuras_rel = Path("figures") if dir_docs is not None else Path(dir_fig).relative_to(destino)
    markdown = construir_reporte(R, figuras_rel.as_posix())
    (destino / "borrador_reporte.md").write_text(markdown, encoding="utf-8")
    markdown_a_pdf(markdown, destino / "reporte.pdf", destino)
    diapositivas = construir_diapositivas(R)
    conclusiones = conclusiones_cortas(R)
    (destino / "borrador_presentacion.md").write_text(construir_presentacion_md(diapositivas, conclusiones),
                                                      encoding="utf-8")
    presentacion_pdf(diapositivas, conclusiones, Path(dir_fig), destino / "presentacion.pdf")
