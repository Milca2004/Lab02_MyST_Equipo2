"""Datos: descarga, congelamiento, carga, auditoría, limpieza y split.

Reglas:
- Se descarga UNA vez con yfinance (diario, auto_adjust=True) y se congela en
  data/prices_daily.csv. Si el CSV existe, nunca se vuelve a descargar,
  porque yfinance puede modificar históricos.
- No se inventan velas: solo se conservan las fechas comunes a los 6 activos.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from src import config

COLUMNAS = ["open", "high", "low", "close", "volume"]


def download_prices(tickers=config.TICKERS, inicio=config.FECHA_INICIO,
                    fin=config.FECHA_FIN):
    """Descarga OHLCV diario ajustado por splits y dividendos (auto_adjust=True).

    Regresa un DataFrame en formato largo: date, ticker, open, high, low, close, volume.
    """
    import yfinance as yf

    # yfinance toma 'end' como exclusivo: se suma un día para incluir FECHA_FIN.
    fin_exclusivo = (pd.Timestamp(fin) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tablas = []
    for ticker in tickers:
        crudo = yf.download(ticker, start=inicio, end=fin_exclusivo,
                            auto_adjust=True, progress=False)
        # Versiones nuevas regresan columnas MultiIndex (campo, ticker).
        if isinstance(crudo.columns, pd.MultiIndex):
            crudo.columns = crudo.columns.get_level_values(0)
        crudo = crudo.rename(columns=str.lower)[COLUMNAS]
        crudo.index.name = "date"
        crudo = crudo.reset_index()
        crudo["ticker"] = ticker
        tablas.append(crudo)
    largo = pd.concat(tablas, ignore_index=True)
    largo["date"] = pd.to_datetime(largo["date"]).dt.tz_localize(None)
    return largo[["date", "ticker"] + COLUMNAS]


def freeze_prices(largo, tickers=config.TICKERS):
    """Guarda el CSV congelado y la metadata de la descarga."""
    import yfinance as yf

    config.DIR_DATA.mkdir(parents=True, exist_ok=True)
    largo.to_csv(config.ARCHIVO_PRECIOS, index=False)
    metadata = {
        "fecha_descarga": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fuente": "Yahoo Finance vía yfinance (auto_adjust=True, diario)",
        "version_yfinance": yf.__version__,
        "tickers": list(tickers),
        "inicio": config.FECHA_INICIO,
        "fin": config.FECHA_FIN,
        "filas": int(len(largo)),
    }
    with open(config.ARCHIVO_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def ensure_data():
    """Descarga y congela solo si el CSV no existe."""
    if config.ARCHIVO_PRECIOS.exists():
        return False
    freeze_prices(download_prices())
    return True


def load_long():
    """Lee el CSV congelado en formato largo."""
    largo = pd.read_csv(config.ARCHIVO_PRECIOS, parse_dates=["date"])
    return largo


def to_panel(largo):
    """Formato largo -> dict {campo: DataFrame(fechas × tickers)}."""
    panel = {}
    for campo in COLUMNAS:
        tabla = largo.pivot_table(index="date", columns="ticker", values=campo,
                                  aggfunc="first")
        panel[campo] = tabla[config.TICKERS].sort_index()
    return panel


def audit_data(largo):
    """Auditoría por ticker contra el calendario común (unión de fechas).

    Revisa: fechas faltantes, NaNs, duplicados, consistencia OHLC,
    precios/volumen <= 0 y rendimientos extremos |r| > 25%.
    """
    calendario = pd.DatetimeIndex(sorted(largo["date"].unique()))
    filas = []
    for ticker in config.TICKERS:
        d = largo[largo["ticker"] == ticker].sort_values("date")
        fechas = pd.DatetimeIndex(d["date"])
        ohlc_max = d[["open", "close"]].max(axis=1)
        ohlc_min = d[["open", "close"]].min(axis=1)
        rend = d["close"].pct_change()
        extremos = d.loc[rend.abs() > config.UMBRAL_RET_EXTREMO, "date"]
        rend_extremos = rend[rend.abs() > config.UMBRAL_RET_EXTREMO]
        detalle = [f"{f.date()}: {r:+.1%}" for f, r in zip(extremos, rend_extremos)]
        filas.append({
            "ticker": ticker,
            "primera_fecha": fechas.min().date(),
            "ultima_fecha": fechas.max().date(),
            "n_filas": len(d),
            "fechas_faltantes": len(calendario.difference(fechas)),
            "nans": int(d[COLUMNAS].isna().sum().sum()),
            "duplicados": int(d["date"].duplicated().sum()),
            "high_menor_que_max_oc": int((d["high"] < ohlc_max - 1e-8).sum()),
            "low_mayor_que_min_oc": int((d["low"] > ohlc_min + 1e-8).sum()),
            "precios_no_positivos": int((d[["open", "high", "low", "close"]] <= 0).sum().sum()),
            "volumen_no_positivo": int((d["volume"] <= 0).sum()),
            "n_rend_extremos": len(detalle),
            "rend_extremos": "; ".join(detalle),
        })
    auditoria = pd.DataFrame(filas)
    inicio_comun = auditoria["primera_fecha"].max()
    fin_comun = auditoria["ultima_fecha"].min()
    auditoria["traslape_completo"] = (auditoria["fechas_faltantes"] == 0)
    auditoria["inicio_comun"] = inicio_comun
    auditoria["fin_comun"] = fin_comun
    return auditoria


def clean_data(panel):
    """Conserva solo las fechas en que los 6 activos tienen vela completa.

    No se rellena nada (sin forward-fill). Regresa (panel_limpio, filas_eliminadas).
    """
    completo = None
    for campo in COLUMNAS:
        ok = panel[campo].notna().all(axis=1)
        completo = ok if completo is None else (completo & ok)
    limpio = {campo: tabla.loc[completo] for campo, tabla in panel.items()}
    filas_eliminadas = int((~completo).sum())
    return limpio, filas_eliminadas


def load_prices():
    """Carga el panel limpio y alineado (dict campo -> fechas × tickers)."""
    panel = to_panel(load_long())
    limpio, _ = clean_data(panel)
    return limpio


def split_train_test(fechas, fraccion=config.FRACCION_TRAIN):
    """Split cronológico sin traslape: primer 80% TRAIN, último 20% TEST."""
    n_train = int(np.floor(len(fechas) * fraccion))
    return {
        "train_inicio": str(fechas[0].date()),
        "train_fin": str(fechas[n_train - 1].date()),
        "test_inicio": str(fechas[n_train].date()),
        "test_fin": str(fechas[-1].date()),
        "n_train": int(n_train),
        "n_test": int(len(fechas) - n_train),
    }


def asset_frame(panel, ticker):
    """OHLCV de un solo activo como DataFrame con columnas open..volume."""
    return pd.DataFrame({campo: panel[campo][ticker] for campo in COLUMNAS})
