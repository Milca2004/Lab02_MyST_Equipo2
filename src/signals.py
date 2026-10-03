"""Indicadores técnicos y regla de confirmación 2 de 3.

Que se hizo 
1. Calcula tres indicadores (EMA, RSI y Bollinger) con los precios diarios de una acción.
2. Cada indicador "vota" cada día: +1 (comprar), -1 (vender) o 0 (no opina).
3. Solo se abre una posición si al menos dos votos van en la misma dirección.

Regla para no mirar al futuro: el valor de un indicador en el día t se calcula
únicamente con datos hasta el cierre de ese día. Por eso solo se usan promedios
que se actualizan día con día (ewm con adjust=False) y ventanas que miran hacia
atrás (rolling). No se permite shift(-k), center=True ni normalizar con toda la
serie, porque cualquiera de esas cosas metería información del futuro.

La operación se ejecuta hasta la apertura del día t+1 (eso lo hace backtest.py).
"""
import numpy as np
import pandas as pd

from src import config

NOMBRES_VOTOS = ["voto_ema", "voto_rsi", "voto_bb"]


def ema(precio, h):
    """Media móvil exponencial: un promedio que da más peso a los días recientes.

    Cada día se mezcla el precio de hoy con el promedio de ayer:
        EMA de hoy = alpha * precio de hoy + (1 - alpha) * EMA de ayer
    con alpha = 2 / (h + 1), donde h es la ventana en días. Con una h chica el
    promedio reacciona rápido; con una h grande, reacciona lento.
    """
    alpha = 2.0 / (h + 1.0)
    return precio.ewm(alpha=alpha, adjust=False).mean()


def wilder(serie, n):
    """Suavizado de Wilder: otro promedio que pesa más lo reciente (alpha = 1/n).

    Se usa dentro del RSI y del ATR. Cada día el valor se mueve un poco hacia el
    dato nuevo:
        valor de hoy = valor de ayer + (dato de hoy - valor de ayer) / n
    """
    return serie.ewm(alpha=1.0 / n, adjust=False).mean()


def rsi(close, n):
    """RSI de Wilder: compara qué tanto ha subido el precio contra qué tanto ha bajado.

    Va de 0 a 100. Paso a paso:
        1. Se calcula el cambio diario del precio.
        2. Las subidas se separan de las bajadas (las bajadas en positivo).
        3. Cada grupo se promedia con el suavizado de Wilder.
        4. RS = promedio de subidas / promedio de bajadas
        5. RSI = 100 - 100 / (1 + RS)
    Un RSI alto significa que ha subido mucho (sobrecompra) y uno bajo que ha
    caído mucho (sobreventa).
    """
    delta = close.diff()
    subida = delta.clip(lower=0)
    bajada = (-delta).clip(lower=0)
    media_subida = wilder(subida, n)
    media_bajada = wilder(bajada, n)
    rs = media_subida / media_bajada
    valor = 100.0 - 100.0 / (1.0 + rs)
    valor[media_bajada == 0] = 100.0          # si no hubo bajadas, el RSI vale 100
    valor.iloc[:n] = np.nan                   # los primeros n días no tienen suficientes datos
    return valor


def bollinger(close, n, k):
    """Bandas de Bollinger: un "tubo" alrededor del precio promedio.

        banda media    = promedio de los últimos n días
        banda superior = banda media + k * desviación estándar de esos n días
        banda inferior = banda media - k * desviación estándar de esos n días
    Si el precio se sale del tubo, está inusualmente alto o bajo. El valor k
    controla qué tan ancho es el tubo. La desviación estándar usa ddof=0
    (divide entre n).
    """
    banda_media = close.rolling(n).mean()
    sigma = close.rolling(n).std(ddof=0)
    banda_sup = banda_media + k * sigma
    banda_inf = banda_media - k * sigma
    return banda_media, banda_sup, banda_inf


def atr(high, low, close, n=config.VENTANA_ATR):
    """ATR: cuánto se mueve el precio en un día típico.

    Sirve para poner el stop-loss y el take-profit. Cada día se toma el mayor
    de tres rangos:
        - máximo del día menos mínimo del día
        - máximo de hoy contra el cierre de ayer (en valor absoluto)
        - mínimo de hoy contra el cierre de ayer (en valor absoluto)
    Ese es el rango verdadero (TR). El ATR es su promedio con suavizado de Wilder.
    """
    cierre_previo = close.shift(1)
    rango1 = high - low
    rango2 = (high - cierre_previo).abs()
    rango3 = (low - cierre_previo).abs()
    tr = pd.concat([rango1, rango2, rango3], axis=1).max(axis=1)
    valor = wilder(tr, n)
    valor.iloc[:n] = np.nan
    return valor


# Los votos de cada indicador: +1 (comprar), -1 (vender) o 0 (no opina)

def voto_ema(close, rapida, lenta):
    """Voto de tendencia: +1 si la EMA rápida está arriba de la lenta, -1 si está abajo.

    Rápida arriba de la lenta quiere decir que el precio viene subiendo.
    """
    diferencia = ema(close, rapida) - ema(close, lenta)
    voto = np.sign(diferencia)
    voto.iloc[:lenta] = 0.0                   # al inicio no hay historia suficiente, no vota
    return voto


def voto_rsi(close, n, umbral_inf, umbral_sup):
    """Voto de momento: +1 si el RSI está debajo del umbral inferior, -1 si está arriba del superior.

    Debajo del inferior es sobreventa (se espera un rebote); arriba del
    superior es sobrecompra. En medio, 0.
    """
    valor = rsi(close, n)
    voto = pd.Series(0.0, index=close.index)
    voto[valor < umbral_inf] = 1.0
    voto[valor > umbral_sup] = -1.0
    return voto


def voto_bollinger(close, n, k):
    """Voto de volatilidad: +1 si el cierre cae debajo de la banda inferior, -1 si sube por encima de la superior.

    Si el cierre está dentro de las bandas, vota 0.
    """
    _, banda_sup, banda_inf = bollinger(close, n, k)
    voto = pd.Series(0.0, index=close.index)
    voto[close < banda_inf] = 1.0
    voto[close > banda_sup] = -1.0
    return voto


def calcular_votos(ohlcv, params):
    """Arma una tabla con los 3 votos de cada día (voto_ema, voto_rsi y voto_bb).

    params trae las ventanas y umbrales de cada indicador. La EMA lenta es la
    rápida más ema_gap días.
    """
    close = ohlcv["close"]
    lenta = params["ema_fast"] + params["ema_gap"]
    votos = pd.DataFrame({
        "voto_ema": voto_ema(close, params["ema_fast"], lenta),
        "voto_rsi": voto_rsi(close, params["rsi_window"], params["rsi_lower"],
                             params["rsi_upper"]),
        "voto_bb": voto_bollinger(close, params["bb_window"], params["bb_k"]),
    })
    return votos


def confirmar(votos, umbral=config.UMBRAL_CONFIRMACION):
    """Regla de confirmación 2 de 3 (de las notas de Risk Parity, paso 7.2).

    Cada día se suman los votos de los k indicadores:
        - Si la suma, sin importar el signo, llega al umbral (2): hay señal y
          s = suma / k.
        - Si no llega: s = 0 y no se abre nada.
    Con 3 indicadores y umbral 2, s solo puede valer -1, -2/3, 0, 2/3 o 1.

    Ejemplos:
        (+1, +1, 0) suma 2, da s = 2/3 y se abre.
        (+1, 0, 0)  suma 1, da s = 0 y no se abre.
        (+1, +1, -1) suma 1, da s = 0 y no se abre: el tercer voto cancela la
        confirmación.
    """
    k = votos.shape[1]
    suma = votos.sum(axis=1)
    s = suma / k
    s[suma.abs() < umbral] = 0.0
    return s


def generate_signals(ohlcv, params, umbral=config.UMBRAL_CONFIRMACION,
                     indicadores=None):
    """Calcula las señales de una acción con los parámetros params.

    Devuelve una tabla con:
        - los 3 votos y su suma
        - s: la fuerza de la señal, entre -1 y 1
        - direccion: +1 compra, -1 vende, 0 no hace nada
        - atr: se usa para el stop-loss y el take-profit

    Los argumentos indicadores y umbral solo se cambian en el experimento de un
    solo indicador (por ejemplo indicadores=["voto_rsi"] con umbral=1). La
    estrategia oficial usa los 3 indicadores con umbral 2.
    """
    votos = calcular_votos(ohlcv, params)
    if indicadores is None:
        indicadores = NOMBRES_VOTOS
    s = confirmar(votos[indicadores], umbral)
    salida = votos.copy()
    salida["suma_votos"] = votos[indicadores].sum(axis=1)
    salida["s"] = s
    salida["direccion"] = np.sign(s)
    salida["atr"] = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    return salida


def correlacion_votos(votos):
    """Correlación entre los 3 votos: qué tanto coinciden entre sí.

    Si es muy alta, los indicadores dicen casi lo mismo y aportan poco por separado.
    """
    return votos[NOMBRES_VOTOS].corr()