"""Figuras (PNG, 150 dpi, en español). Solo grafican: leen las tablas ya calculadas por los
demás módulos (en docs/resultados/)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src import config  # noqa: E402

# Paleta categórica (orden fijo, nunca se cicla)
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRIS = "#52514e"
COLOR_ACTIVO = {a: COLORES[i] for i, a in enumerate(config.TICKERS)}
COLOR_REGIMEN = {0: COLORES[0], 1: COLORES[1], 2: COLORES[2]}
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])

# Curvas que se comparan en las figuras de valor y drawdown: (color, estilo de línea)
CURVAS = {"RP (sistema)": (COLORES[0], "-"), "Pesos iguales (EW)": (COLORES[1], "--"),
          "Buy & Hold EW": (GRIS, ":")}

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.8, "lines.linewidth": 2.0,
    "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
})

# Nombre de cada conjunto en títulos (los archivos conservan el sufijo interno)
ETIQUETA = {"wf_oos": "WF-OOS", "test": "TEST"}


def _guardar(fig, dir_fig, nombre):
    fig.tight_layout()
    fig.savefig(dir_fig / nombre, bbox_inches="tight")
    plt.close(fig)


def _leer(dir_res, nombre, **kw):
    ruta = dir_res / nombre
    return pd.read_csv(ruta, **kw) if ruta.exists() else None


def _paneles(curvas_oos, curvas_test):
    """Un panel para TRAIN y, si ya se corrió, otro para TEST."""
    paneles = [("TRAIN · WF-OOS", curvas_oos)]
    if curvas_test is not None:
        paneles.append(("TEST", curvas_test))
    return paneles


def _sombrear_regimenes(ax, regimen):
    """Sombrea el fondo según el régimen vigente."""
    regimen = regimen[regimen >= 0]
    if regimen.empty:
        return
    inicio, actual = regimen.index[0], regimen.iloc[0]
    for fecha, valor in regimen.iloc[1:].items():
        if valor != actual:
            ax.axvspan(inicio, fecha, color=COLOR_REGIMEN[int(actual)], alpha=0.13, lw=0)
            inicio, actual = fecha, valor
    ax.axvspan(inicio, regimen.index[-1], color=COLOR_REGIMEN[int(actual)], alpha=0.13, lw=0)


def _leyenda_regimenes(ax, loc="upper left"):
    parches = [Patch(color=COLOR_REGIMEN[j], alpha=0.35, label=config.REGIMENES[j])
               for j in range(3)]
    return ax.legend(handles=parches, loc=loc, title="Régimen")


# --- 1 a 3: valor, drawdown y rendimientos ------------------------------------
def fig_valor_portafolio(curvas_oos, curvas_test, dir_fig):
    paneles = _paneles(curvas_oos, curvas_test)
    fig, ejes = plt.subplots(1, len(paneles), figsize=(8 * len(paneles), 6), squeeze=False)
    for ax, (titulo, curvas) in zip(ejes[0], paneles):
        for c, (color, estilo) in CURVAS.items():
            ax.plot(curvas.index, curvas[c] / 1e6, color=color, ls=estilo, label=c)
        ax.set_yscale("log")
        ax.set_title(f"Valor del portafolio — {titulo}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Valor (millones USD, escala log)")
        ax.legend(loc="upper left")
    _guardar(fig, dir_fig, "01_valor_portafolio.png")


def fig_drawdown(curvas_oos, curvas_test, dir_fig):
    paneles = _paneles(curvas_oos, curvas_test)
    fig, ejes = plt.subplots(1, len(paneles), figsize=(8 * len(paneles), 5), squeeze=False)
    for ax, (titulo, curvas) in zip(ejes[0], paneles):
        for c, (color, _) in CURVAS.items():
            dd = (curvas[c] - curvas[c].cummax()) / curvas[c].cummax()
            ax.plot(dd.index, dd * 100, color=color, label=c)
        ax.set_title(f"Drawdown — {titulo}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Drawdown (%)")
        ax.legend(loc="lower left")
    _guardar(fig, dir_fig, "02_drawdown.png")


def fig_rendimientos(mensual, anual, conjunto, dir_fig):
    serie = mensual["RP (sistema)"]
    tabla = pd.DataFrame({"anio": serie.index.year, "mes": serie.index.month,
                          "r": serie.to_numpy() * 100})
    matriz = tabla.pivot(index="anio", columns="mes", values="r")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 0.5 * len(matriz) + 3),
                                   gridspec_kw={"width_ratios": [3, 1.2]})
    limite = max(1e-9, np.nanmax(np.abs(matriz.to_numpy())))
    imagen = ax1.imshow(matriz.to_numpy(), cmap=DIVERGENTE, vmin=-limite, vmax=limite,
                        aspect="auto")
    ax1.set_xticks(range(matriz.shape[1]), [str(m) for m in matriz.columns])
    ax1.set_yticks(range(len(matriz)), [str(a) for a in matriz.index])
    ax1.set_xlabel("Mes")
    ax1.set_ylabel("Año")
    ax1.set_title(f"Rendimiento mensual del sistema RP (%) — {ETIQUETA[conjunto]}")
    ax1.grid(False)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            v = matriz.iloc[i, j]
            if np.isfinite(v):
                ax1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="#0b0b0b")
    fig.colorbar(imagen, ax=ax1, label="%")

    anios = anual.index.year
    ancho = 0.27
    for k, (col, color) in enumerate([("RP (sistema)", COLORES[0]), ("EW", COLORES[1]),
                                      ("Buy & Hold", GRIS)]):
        ax2.barh(np.arange(len(anios)) + (k - 1) * ancho, anual[col] * 100, height=ancho,
                 color=color, label=col)
    ax2.set_yticks(range(len(anios)), [str(a) for a in anios])
    ax2.invert_yaxis()
    ax2.set_xlabel("Rendimiento anual (%)")
    ax2.set_title("Rendimiento anual")
    ax2.legend(loc="lower right", fontsize=10)
    _guardar(fig, dir_fig, f"03_rendimientos_{conjunto}.png")


# --- 4 y 5: sensibilidad y costos ---------------------------------------------
def fig_sensibilidad(sens, dir_fig):
    parametros = list(dict.fromkeys(sens["parametro"]))
    columnas = 5
    filas = int(np.ceil(len(parametros) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(4.2 * columnas, 3.6 * filas), sharey=True)
    base = sens[sens["factor"] == 0]["calmar"].iloc[0]
    for ax, p in zip(ejes.flat, parametros):
        d = sens[sens["parametro"] == p].sort_values("factor")
        ax.plot(d["factor"] * 100, d["calmar"], marker="o", ms=8, color=COLORES[0])
        ax.axhline(base, color=GRIS, ls=":", lw=1.2)
        ax.set_title(p, fontsize=14)
        ax.set_xticks([-20, -10, 0, 10, 20])
    for ax in ejes.flat[len(parametros):]:
        ax.axis("off")
    for ax in ejes[:, 0]:
        ax.set_ylabel("Calmar (TRAIN)")
    for ax in ejes[-1, :]:
        ax.set_xlabel("Cambio del parámetro (%)")
    fig.suptitle("Sensibilidad ±20% (uno a la vez, θ congelado como base; línea punteada = base)",
                 fontsize=17)
    _guardar(fig, dir_fig, "04_sensibilidad.png")


def fig_curva_costos(curva, resumen, dir_fig):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curva["comision"] * 100, curva["retorno_neto"] * 100, marker="o", ms=6,
            color=COLORES[0], label="Retorno neto acumulado (WF-OOS)")
    ax.axhline(0, color=GRIS, lw=1)
    ax.axvline(config.COMISION * 100, color=COLORES[1], ls="--", label="Comisión oficial 0.125%")
    equilibrio = resumen.get("comision_equilibrio")
    if equilibrio is not None and np.isfinite(equilibrio):
        ax.axvline(equilibrio * 100, color=COLORES[2], ls=":",
                   label=f"Punto de equilibrio {equilibrio * 100:.3f}%")
    ax.set_xlabel("Comisión por lado (%)")
    ax.set_ylabel("Retorno neto acumulado (%)")
    ax.set_title("Retorno neto vs. nivel de costo")
    ax.legend()
    _guardar(fig, dir_fig, "05_curva_costos.png")


# --- 6: régimen ---------------------------------------------------------------
def fig_regimenes(serie, curvas_oos, dir_fig):
    regimen = serie["regimen"]
    fig, ax = plt.subplots(figsize=(14, 6))
    _sombrear_regimenes(ax, regimen)
    ax.plot(serie.index, serie["indice_ew"], color="#0b0b0b", lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Régimen confirmado (WF-OOS) sobre el índice equiponderado")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Índice EW (base 1, log)")
    _leyenda_regimenes(ax)
    _guardar(fig, dir_fig, "06a_regimenes_linea_tiempo.png")

    from src.regimes import FEATURES
    validos = serie[serie["regimen"] >= 0]
    fig, ejes = plt.subplots(1, len(FEATURES), figsize=(4.2 * len(FEATURES), 4.8))
    for ax, f in zip(ejes, FEATURES):
        datos = [validos.loc[validos["regimen"] == j, f].dropna() for j in range(3)]
        caja = ax.boxplot(datos, patch_artist=True, widths=0.6, showfliers=False)
        for parche, j in zip(caja["boxes"], range(3)):
            parche.set_facecolor(COLOR_REGIMEN[j])
            parche.set_alpha(0.6)
        ax.set_xticks([1, 2, 3], ["Tend.", "Rev.", "Crisis"])
        ax.set_title(f, fontsize=14)
    ejes[0].set_ylabel("Valor de la feature")
    fig.suptitle("Distribución de las features por régimen (WF-OOS)", fontsize=17)
    _guardar(fig, dir_fig, "06b_features_por_regimen.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    _sombrear_regimenes(ax, regimen.reindex(curvas_oos.index))
    ax.plot(curvas_oos.index, curvas_oos["RP (sistema)"] / 1e6, color=COLORES[0])
    ax.set_title("Valor del sistema RP con regímenes sombreados (WF-OOS)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor (millones USD)")
    _leyenda_regimenes(ax)
    _guardar(fig, dir_fig, "06c_valor_con_regimenes.png")


def fig_transiciones(eventos, persistencia, dir_fig):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(3)
    ax1.bar(x - 0.2, persistencia["duracion_promedio_obs"], width=0.4, color=COLORES[0],
            label="Observada")
    ax1.bar(x + 0.2, persistencia["duracion_esperada_markov"], width=0.4, color=COLORES[1],
            label="Esperada 1/(1−A_jj)")
    ax1.axhline(10, color=GRIS, ls=":", label="Objetivo ≥ 10 días")
    ax1.set_xticks(x, persistencia["regimen"])
    ax1.set_ylabel("Días hábiles")
    ax1.set_title("Duración promedio por régimen")
    ax1.legend()
    if len(eventos):
        etiquetas = [f"{config.REGIMENES[int(a)][:4]}→{config.REGIMENES[int(b)][:4]}"
                     for a, b in zip(eventos["de"], eventos["a"]) if a >= 0]
        conteo = pd.Series(etiquetas).value_counts().sort_index()
        ax2.bar(conteo.index, conteo.values, color=COLORES[0])
        ax2.tick_params(axis="x", rotation=30)
    ax2.set_ylabel("Número de transiciones")
    ax2.set_title("Transiciones de régimen (WF-OOS)")
    _guardar(fig, dir_fig, "06d_transiciones.png")


# --- 7: portafolio ------------------------------------------------------------
def fig_contribuciones(promedio, dir_fig):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(config.TICKERS))
    nombres = {"ew": "Pesos iguales", "naive": "RP naive", "rp": "RP Spinu"}
    for k, metodo in enumerate(nombres):
        d = promedio[promedio["metodo"] == metodo].set_index("activo").loc[config.TICKERS]
        ax.bar(x + (k - 1) * 0.27, d["rc_pct"] * 100, width=0.25, color=COLORES[k],
               label=nombres[metodo])
    ax.axhline(100 / 6, color=GRIS, ls=":", label="1/n = 16.7%")
    ax.set_xticks(x, config.TICKERS)
    ax.set_ylabel("% del riesgo total (RC_i/σ_p)")
    ax.set_title("Contribución al riesgo por activo (promedio WF-OOS)")
    ax.legend()
    _guardar(fig, dir_fig, "07a_contribuciones_riesgo.png")


def fig_senales(senales, dir_fig, conjunto):
    semanal = senales.resample("W").mean()
    fig, ax = plt.subplots(figsize=(15, 4.5))
    imagen = ax.imshow(semanal.T.to_numpy(), aspect="auto", cmap=DIVERGENTE, vmin=-1, vmax=1,
                       interpolation="nearest")
    ax.set_yticks(range(len(semanal.columns)), semanal.columns)
    pasos = max(1, len(semanal) // 10)
    ax.set_xticks(range(0, len(semanal), pasos),
                  [d.strftime("%Y-%m") for d in semanal.index[::pasos]], rotation=30)
    ax.grid(False)
    ax.set_xlabel("Semana")
    ax.set_title(f"Fuerza de señal s_i por activo (promedio semanal) — {ETIQUETA[conjunto]}")
    fig.colorbar(imagen, ax=ax, label="s_i (−1 corto … +1 largo)")
    _guardar(fig, dir_fig, f"07b_senales_s_{conjunto}.png")


def fig_correlacion_regimenes(matrices, dir_fig):
    regimenes = list(dict.fromkeys(matrices["regimen"]))
    fig, ejes = plt.subplots(1, len(regimenes), figsize=(6 * len(regimenes), 5.5))
    ejes = np.atleast_1d(ejes)
    for ax, r in zip(ejes, regimenes):
        m = matrices[matrices["regimen"] == r].drop(columns="regimen").set_index("ticker")
        imagen = ax.imshow(m.to_numpy(), cmap=SECUENCIAL, vmin=0, vmax=1)
        ax.set_xticks(range(len(m)), m.columns, rotation=45)
        ax.set_yticks(range(len(m)), m.index)
        ax.grid(False)
        for i in range(len(m)):
            for j in range(len(m)):
                valor = m.iloc[i, j]
                ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if valor > 0.6 else "#0b0b0b")
        ax.set_title(f"Correlación — {r}")
    fig.colorbar(imagen, ax=ejes.tolist(), label="Correlación", shrink=0.8)
    fig.savefig(dir_fig / "07c_correlacion_por_regimen.png", bbox_inches="tight")
    plt.close(fig)


def fig_barrido_rebalanceo(barrido, elegido, dir_fig):
    metricas = [("retorno_bruto", "Retorno bruto (%)"), ("costo_total", "Costo total (%)"),
                ("retorno_neto", "Retorno neto (%)"), ("turnover_anual", "Turnover anual (×)")]
    fig, ejes = plt.subplots(1, 4, figsize=(22, 5.5))
    for ax, (col, etiqueta) in zip(ejes, metricas):
        factor = 1 if col == "turnover_anual" else 100
        for k, f in enumerate(config.FRECUENCIAS_REBALANCEO):
            d = barrido[barrido["frecuencia"] == f]
            ax.plot(d["delta"], d[col] * factor, marker="o", ms=8, color=COLORES[k], label=f)
        ax.set_xlabel("Banda δ")
        ax.set_title(etiqueta)
    ejes[0].legend(title="Frecuencia f")
    fig.suptitle(f"Barrido de rebalanceo f × δ en TRAIN (WF-OOS). Elegido: {elegido['frecuencia']}, "
                 f"δ = {elegido['delta']:.2f}", fontsize=17)
    _guardar(fig, dir_fig, "07d_barrido_rebalanceo.png")


def fig_pesos_estimadores(pesos, dir_fig):
    estimadores = list(dict.fromkeys(pesos["estimador"]))
    fig, ejes = plt.subplots(len(estimadores), 1, figsize=(14, 3.6 * len(estimadores)),
                             sharex=True)
    titulos = {"muestral": "Muestral (504 días, oficial)", "ewma": "EWMA λ = 0.94",
               "ledoit_wolf": "Ledoit-Wolf"}
    for ax, e in zip(ejes, estimadores):
        d = pesos[pesos["estimador"] == e].set_index("date")
        for a in config.TICKERS:
            ax.plot(d.index, d[a], color=COLOR_ACTIVO[a], lw=1.5, label=a)
        ax.set_title(f"Pesos RP — {titulos.get(e, e)}")
        ax.set_ylabel("Peso")
    ejes[0].legend(ncol=6, loc="upper left", fontsize=11)
    ejes[-1].set_xlabel("Fecha")
    _guardar(fig, dir_fig, "08_pesos_estimadores_sigma.png")


# --- Adicionales --------------------------------------------------------------
def fig_un_indicador(tabla, dir_fig):
    port = tabla[tabla["nivel"] == "portafolio"]
    colores = [COLORES[0] if c == "2 de 3" else "#86b6ef" for c in port["caso"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax1.bar(port["caso"], port["n_operaciones"], color=colores)
    ax1.set_ylabel("Número de operaciones")
    ax1.set_title("Operaciones (portafolio, WF-OOS)")
    ax2.bar(port["caso"], port["calmar"], color=colores)
    ax2.axhline(0, color=GRIS, lw=1)
    ax2.set_ylabel("Calmar")
    ax2.set_title("Calmar (portafolio, WF-OOS)")
    for ax in (ax1, ax2):
        ax.tick_params(axis="x", rotation=15)
    _guardar(fig, dir_fig, "09_dos_de_tres_vs_un_indicador.png")


def fig_individuales(curvas, dir_fig, conjunto):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.plot(curvas.index, curvas["RP (sistema)"] / 1e6, color="#0b0b0b", lw=2.6,
            label="Portafolio RP")
    for a in config.TICKERS:
        ax.plot(curvas.index, curvas[f"{a} (estrategia sola)"] / 1e6, color=COLOR_ACTIVO[a],
                lw=1.4, label=f"{a} sola")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor (millones USD)")
    ax.set_title(f"Portafolio vs. estrategia en cada activo (100% del capital) — {ETIQUETA[conjunto]}")
    ax.legend(ncol=4, fontsize=11)
    _guardar(fig, dir_fig, f"10_portafolio_vs_individuales_{conjunto}.png")


def fig_costos_vs_bruto(metricas_tabla, dir_fig, conjunto):
    d = metricas_tabla.dropna(subset=["costos_totales"])
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(d))
    ax.bar(x - 0.2, d["pnl_bruto"] / 1e3, width=0.4, color=COLORES[0], label="PnL bruto")
    ax.bar(x + 0.2, d["costos_totales"] / 1e3, width=0.4, color=COLORES[1],
           label="Costos totales")
    ax.axhline(0, color=GRIS, lw=1)
    ax.set_xticks(x, [n.replace(" (estrategia sola)", "") for n in d["nombre"]], rotation=20)
    ax.set_ylabel("Miles de USD")
    ax.set_title(f"Costos totales vs. retorno bruto — {ETIQUETA[conjunto]}")
    ax.legend()
    _guardar(fig, dir_fig, f"11_costos_vs_bruto_{conjunto}.png")


def fig_rolling_anchored(curvas, dir_fig):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(curvas.index, curvas["RP (sistema)"] / 1e6, color=COLORES[0], label="Rolling (oficial)")
    for c in [c for c in curvas.columns if c.startswith("anchored")]:
        ax.plot(curvas.index, curvas[c] / 1e6, color=COLORES[1], ls="--", label=c.capitalize())
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Valor (millones USD)")
    ax.set_title("Walk-forward rolling vs. anchored (WF-OOS)")
    ax.legend()
    _guardar(fig, dir_fig, "12_rolling_vs_anchored.png")


def fig_correlacion_senales(corr, dir_fig):
    orden = ["voto_ema", "voto_rsi", "voto_bb"]
    media = corr.groupby(["voto_a", "voto_b"])["correlacion"].mean().unstack().loc[orden, orden]
    nombres = ["EMA", "RSI", "Bollinger"]
    fig, ax = plt.subplots(figsize=(7, 6))
    imagen = ax.imshow(media.to_numpy(), cmap=DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(3), nombres)
    ax.set_yticks(range(3), nombres)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{media.iloc[i, j]:.2f}", ha="center", va="center", fontsize=14)
    ax.set_title("Correlación entre votos (promedio 6 activos)")
    fig.colorbar(imagen, ax=ax)
    _guardar(fig, dir_fig, "13_correlacion_senales.png")


def fig_optuna(trials, importancia, superficie, info, dir_fig):
    validos = trials[trials["value"] > config.OBJETIVO_INVALIDO]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(validos["number"], validos["value"], s=40, color=COLORES[0], label="Trial válido")
    mejor = trials["value"].where(trials["value"] > config.OBJETIVO_INVALIDO).cummax()
    ax.plot(trials["number"], mejor, color=COLORES[1], label="Mejor hasta el trial")
    ax.axvline(config.N_STARTUP_TRIALS - 0.5, color=GRIS, ls=":", label="Fin de la fase aleatoria")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Calmar (train)")
    ax.set_title(f"Historia de optimización — {info['activo']}, "
                 f"ventana {info['train_ini']} a {info['train_fin']}")
    ax.legend()
    _guardar(fig, dir_fig, "14a_optuna_historia.png")

    if importancia is not None and len(importancia):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        d = importancia.sort_values("importancia")
        ax.barh(d["parametro"], d["importancia"], color=COLORES[0])
        ax.set_xlabel("Importancia (fANOVA)")
        ax.set_title("Importancia de parámetros")
        _guardar(fig, dir_fig, "14b_optuna_importancia.png")

    columnas = [c for c in trials.columns if c.startswith("params_")]
    fig, ejes = plt.subplots(2, 5, figsize=(21, 8), sharey=True)
    for ax, c in zip(ejes.flat, columnas):
        ax.scatter(validos[c], validos["value"], s=30, color=COLORES[0], alpha=0.8)
        ax.set_title(c.replace("params_", ""), fontsize=14)
    for ax in ejes[:, 0]:
        ax.set_ylabel("Calmar")
    fig.suptitle("Slice plots: Calmar vs. cada parámetro (trials válidos)", fontsize=17)
    _guardar(fig, dir_fig, "14c_optuna_slices.png")

    if superficie is not None and len(superficie):
        p1, p2 = info["parametros_superficie"]
        tabla = superficie.pivot(index=p2, columns=p1, values="calmar")
        validos_z = np.isfinite(tabla.to_numpy())
        # plot_surface solo dibuja celdas con sus 4 esquinas válidas; sin ninguna, la figura sale vacía
        hay_superficie = (validos_z[1:, 1:] & validos_z[:-1, 1:] & validos_z[1:, :-1] & validos_z[:-1, :-1]).any()
        if hay_superficie:
            X, Y = np.meshgrid(tabla.columns.to_numpy(dtype=float), tabla.index.to_numpy(dtype=float))
            fig = plt.figure(figsize=(11, 8))
            ax = fig.add_subplot(projection="3d")
            ax.plot_surface(X, Y, np.ma.masked_invalid(tabla.to_numpy()), cmap=SECUENCIAL,
                            edgecolor="#ffffff", linewidth=0.3)
            ax.set_xlabel(p1, labelpad=12)
            ax.set_ylabel(p2, labelpad=12)
            ax.set_zlabel("Calmar", labelpad=8)
            ax.set_title("Superficie del Calmar: corte 2D de un espacio de 10 dimensiones\n"
                         "(resto de parámetros fijos en el θ robusto; huecos = < N_MIN operaciones)",
                         fontsize=14)
        else:
            # Los puntos válidos no forman superficie: mapa 2D de la malla completa
            ops = superficie.pivot(index=p2, columns=p1, values="n_operaciones")
            fig, ax = plt.subplots(figsize=(12, 7))
            imagen = ax.imshow(ops.to_numpy(), cmap=SECUENCIAL, aspect="auto", origin="lower")
            ax.set_xticks(range(len(ops.columns)), [f"{v:g}" for v in ops.columns])
            ax.set_yticks(range(len(ops.index)), [f"{v:.2f}" for v in ops.index])
            ax.grid(False)
            for i in range(tabla.shape[0]):
                for j in range(tabla.shape[1]):
                    z = tabla.iloc[i, j]
                    if np.isfinite(z):
                        ax.text(j, i, f"{z:.2f}", ha="center", va="center", fontsize=10,
                                color="white", weight="bold")
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            fig.colorbar(imagen, ax=ax, label="Número de operaciones en train")
            ax.set_title(f"Corte {p1} × {p2} (resto en el θ robusto): solo {int(validos_z.sum())} de "
                         f"{validos_z.size} puntos\nalcanzan N_MIN operaciones; el número en la celda es "
                         "su Calmar (sin superficie que dibujar)", fontsize=14)
            fig.tight_layout()
        fig.savefig(dir_fig / "14d_optuna_superficie_3d.png", bbox_inches="tight")
        plt.close(fig)


def fig_ic_regimenes(por_regimen, dir_fig):
    """Media diaria del sistema RP por régimen con su IC 95% bootstrap (WF-OOS y TEST)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    paneles = [(c, d[d["nombre"] == "RP (sistema)"]) for c, d in por_regimen.items() if d is not None]
    for k, (conjunto, d) in enumerate(paneles):
        y = np.arange(len(d)) + (k - (len(paneles) - 1) / 2) * 0.25
        media = d["media_diaria"].to_numpy() * 1e4
        error = [media - d["ic95_inf"].to_numpy() * 1e4, d["ic95_sup"].to_numpy() * 1e4 - media]
        ax.errorbar(media, y, xerr=error, fmt="o", ms=9, capsize=6, color=COLORES[k],
                    label=ETIQUETA[conjunto])
    ax.axvline(0, color=GRIS, ls=":", lw=1.2)
    ax.set_yticks(range(len(paneles[0][1])), paneles[0][1]["regimen"])
    ax.invert_yaxis()
    ax.set_xlabel("Rendimiento diario medio del sistema RP (pb) e IC 95% bootstrap")
    ax.set_title("¿Difiere el desempeño entre regímenes? (IC que cruza 0 = no significativo)")
    ax.legend(loc="lower right")
    _guardar(fig, dir_fig, "15_ic_por_regimen.png")


# ------------------------------------------------------------------------------
def generar_todas(dir_res, dir_fig):
    """Genera todas las figuras que tengan sus datos disponibles."""
    curvas_oos = _leer(dir_res, "equity_wf_oos.csv", index_col=0, parse_dates=True)
    curvas_test = _leer(dir_res, "equity_test.csv", index_col=0, parse_dates=True)
    fig_valor_portafolio(curvas_oos, curvas_test, dir_fig)
    fig_drawdown(curvas_oos, curvas_test, dir_fig)
    for conjunto, curvas in [("wf_oos", curvas_oos), ("test", curvas_test)]:
        mensual = _leer(dir_res, f"retornos_mensual_{conjunto}.csv", index_col=0, parse_dates=True)
        anual = _leer(dir_res, f"retornos_anual_{conjunto}.csv", index_col=0, parse_dates=True)
        if mensual is not None:
            fig_rendimientos(mensual, anual, conjunto, dir_fig)
        if curvas is not None:
            fig_individuales(curvas, dir_fig, conjunto)
        tabla = _leer(dir_res, f"metricas_{conjunto}.csv")
        if tabla is not None:
            fig_costos_vs_bruto(tabla, dir_fig, conjunto)
        senales = _leer(dir_res, f"senales_s_{conjunto}.csv", index_col=0, parse_dates=True)
        if senales is not None:
            fig_senales(senales, dir_fig, conjunto)
    fig_sensibilidad(_leer(dir_res, "sensibilidad.csv"), dir_fig)
    with open(dir_res / "costos_resumen.json", encoding="utf-8") as f:
        fig_curva_costos(_leer(dir_res, "curva_costos.csv"), json.load(f), dir_fig)
    serie = _leer(dir_res, "regimen_serie.csv", index_col=0, parse_dates=True)
    fig_regimenes(serie.loc[curvas_oos.index[0]:], curvas_oos, dir_fig)
    fig_transiciones(_leer(dir_res, "eventos_regimen_wf_oos.csv"),
                     _leer(dir_res, "regimen_persistencia_wf_oos.csv"), dir_fig)
    fig_ic_regimenes({c: _leer(dir_res, f"metricas_por_regimen_{c}.csv") for c in ("wf_oos", "test")},
                     dir_fig)
    fig_contribuciones(_leer(dir_res, "contribuciones_riesgo_promedio.csv"), dir_fig)
    fig_correlacion_regimenes(_leer(dir_res, "correlacion_por_regimen_matrices.csv"), dir_fig)
    with open(dir_res / "theta_congelado.json", encoding="utf-8") as f:
        elegido = json.load(f)["rebalanceo"]
    fig_barrido_rebalanceo(_leer(dir_res, "rebalanceo_barrido.csv"), elegido, dir_fig)
    fig_pesos_estimadores(_leer(dir_res, "pesos_estimadores.csv", parse_dates=["date"]), dir_fig)
    fig_un_indicador(_leer(dir_res, "un_indicador.csv"), dir_fig)
    fig_rolling_anchored(curvas_oos, dir_fig)
    fig_correlacion_senales(_leer(dir_res, "correlacion_senales.csv"), dir_fig)
    ruta_info = dir_res / "optuna_diagnostico.json"
    if ruta_info.exists():
        with open(ruta_info, encoding="utf-8") as f:
            info = json.load(f)
        fig_optuna(_leer(dir_res, "optuna_trials_diagnostico.csv"),
                   _leer(dir_res, "optuna_importancia.csv"),
                   _leer(dir_res, "optuna_superficie.csv"), info, dir_fig)
