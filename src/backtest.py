"""Motor de backtesting event-driven (un solo motor para 1 o n activos).

Orden de eventos en el día t:
  1. OPEN de t: primero se revisan gaps (si el open ya rebasó el SL o el TP de
     una posición previa, sale al open); luego se ejecutan las órdenes
     pendientes decididas al cierre de t−1 (salidas, entradas, rebalanceos)
     al precio de apertura, cobrando costos. Se valida que no haya
     apalancamiento.
  2. DURANTE t: se revisa SL/TP con high/low, incluida la barra de entrada.
     Si SL y TP caen en la misma barra se ejecuta PRIMERO EL SL (convención
     conservadora).
  3. CIERRE de t: se cobra el borrow fee de cortos, se marca a mercado, se
     registra el equity y, con datos ≤ t, se deciden las órdenes para el
     open de t+1.

Flujos de efectivo (c = tasa de costo por lado):
  compra de q a P:  Cash −= q·P·(1 + c)        venta de q a P:  Cash += q·P·(1 − c)
  un corto abre vendiendo (Cash += q·P·(1 − c)) y queda el pasivo q < 0.
  Equity = Cash + Σ_i q_i·P_i  (q_i < 0 en cortos).

Sin apalancamiento: antes de ejecutar, si Σ|q_i·P_i| > Equity/(1 + c), todas
las posiciones se escalan hacia abajo. Con exposición bruta ≤ 100% el margen
de cortos (50% inicial, 25–30% de mantenimiento) siempre se cumple.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.portfolio import pesos_objetivo

LARGO, CORTO = 1, -1


@dataclass
class Costos:
    """Componentes del costo de transacción.

    Oficial: solo la comisión (0.125% por lado). Los demás valen 0 salvo en el
    escenario de costos realistas.
      spread_bps:       spread bid-ask completo; se paga la mitad en cada lado.
      slippage_bps:     deslizamiento adicional por lado.
      impacto_eta:      impacto = η·|q/ADV|^(2/3) como fracción del precio.
      borrow_fee_anual: costo anual sobre el nocional en corto (diario /252).
    """
    comision: float = config.COMISION
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impacto_eta: float = 0.0
    borrow_fee_anual: float = 0.0


@dataclass
class DatosMotor:
    """Insumos del motor; todos los arreglos son (T días × n activos).

    s:        fuerza de señal s_i calculada al cierre de t (θ vigente en t).
    atr:      ATR al cierre de t (para fijar SL/TP al entrar).
    m_sl, m_tp, max_hold: parámetros θ vigentes en t para entradas nuevas.
    entry_mask: True si se permite abrir en t (optimización por régimen).
    regimen:  (T,) régimen confirmado al cierre de t (None = sin capa de régimen).
    w_rp:     pesos de largo plazo vigentes en t (None = peso 1 por activo).
    corr:     (T × n × n) correlaciones para la política de conflictos (None = sin).
    revision: (T,) fechas de revisión del rebalanceo (None = nunca).
    """
    fechas: pd.DatetimeIndex
    activos: list
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    s: np.ndarray
    atr: np.ndarray
    m_sl: np.ndarray
    m_tp: np.ndarray
    max_hold: np.ndarray
    entry_mask: np.ndarray = None
    volumen: np.ndarray = None
    regimen: np.ndarray = None
    w_rp: np.ndarray = None
    corr: np.ndarray = None
    revision: np.ndarray = None
    delta: float = config.DELTA_DEFAULT
    m_regimen: dict = None        # m(régimen); None = config.M_REGIMEN
    cerrar_al_final: bool = False
    capital: float = config.CAPITAL_INICIAL


@dataclass
class Posicion:
    """Estado de una posición abierta (los niveles SL/TP quedan fijos al entrar)."""
    cantidad: float
    direccion: int
    precio_entrada: float
    idx_entrada: int
    nivel_sl: float
    nivel_tp: float
    s_entrada: float
    max_hold: int
    regimen_entrada: int
    flujo_bruto: float = 0.0     # Σ (−dq·P): al cerrar es el PnL bruto
    costos: float = 0.0


@dataclass
class Orden:
    """Orden pendiente para el open de t+1 de un activo."""
    peso: float                   # peso objetivo (0 = cerrar)
    motivo_salida: str = None     # si la orden cierra la posición actual
    entrada: dict = None          # si la orden abre una posición nueva


@dataclass
class ResultadoBacktest:
    equity: pd.Series
    efectivo: pd.Series
    posiciones: pd.DataFrame
    pesos: pd.DataFrame
    operaciones: pd.DataFrame
    costos_diarios: pd.Series
    nocional_operado: pd.Series
    rebalanceos: pd.DataFrame
    eventos_regimen: pd.DataFrame
    conflictos: int
    exposicion_bruta_max: float

    @property
    def costos_totales(self):
        return float(self.costos_diarios.sum())

    @property
    def turnover_anual(self):
        """Nocional operado anual / equity promedio."""
        anios = len(self.equity) / config.DIAS_ANIO
        return float(self.nocional_operado.sum() / self.equity.mean() / anios)

    @property
    def n_operaciones(self):
        return len(self.operaciones)


class _Motor:
    """Estado mutable del backtest. Cada método es un paso del día."""

    def __init__(self, datos, costos):
        self.d = datos
        self.c = costos
        T, n = datos.close.shape
        self.T, self.n = T, n
        self.cash = datos.capital
        self.pos = [None] * n
        self.pendientes = {}
        self.operaciones = []
        self.rebalanceos = []
        self.eventos_regimen = []
        self.conflictos = 0
        self.costo_dia = 0.0
        self.nocional_dia = 0.0
        self.adv = self._calcular_adv()

    # ------------------------------------------------------------------ costos
    def _calcular_adv(self):
        """Volumen promedio de 20 días hasta t−1 (para el impacto de mercado)."""
        if self.d.volumen is None or self.c.impacto_eta == 0:
            return None
        vol = pd.DataFrame(self.d.volumen)
        return vol.rolling(config.VENTANA_ADV, min_periods=1).mean().shift(1).to_numpy()

    def _tasa_costo(self, i, t, dq):
        """Costo por unidad de nocional de una operación de dq acciones."""
        tasa = self.c.comision + self.c.spread_bps / 2e4 + self.c.slippage_bps / 1e4
        if self.adv is not None and np.isfinite(self.adv[t, i]) and self.adv[t, i] > 0:
            tasa += self.c.impacto_eta * (abs(dq) / self.adv[t, i]) ** (2.0 / 3.0)
        return tasa

    def _operar(self, i, t, dq, precio):
        """Ejecuta dq acciones (con signo) del activo i y cobra el costo."""
        nocional = abs(dq) * precio
        costo = nocional * self._tasa_costo(i, t, dq)
        self.cash -= dq * precio      # compra resta, venta suma
        self.cash -= costo
        self.costo_dia += costo
        self.nocional_dia += nocional
        posicion = self.pos[i]
        posicion.flujo_bruto -= dq * precio
        posicion.costos += costo
        posicion.cantidad += dq

    # --------------------------------------------------------- abrir / cerrar
    def _abrir(self, i, t, cantidad, info):
        precio_entrada = self.d.open[t, i]
        direccion = info["direccion"]
        distancia_sl = info["m_sl"] * info["atr"]
        distancia_tp = info["m_tp"] * info["atr"]
        self.pos[i] = Posicion(
            cantidad=0.0, direccion=direccion, precio_entrada=precio_entrada,
            idx_entrada=t,
            nivel_sl=precio_entrada - direccion * distancia_sl,
            nivel_tp=precio_entrada + direccion * distancia_tp,
            s_entrada=info["s"], max_hold=info["max_hold"],
            regimen_entrada=info["regimen"])
        self._operar(i, t, cantidad, precio_entrada)

    def _cerrar(self, i, t, precio, motivo):
        posicion = self.pos[i]
        self._operar(i, t, -posicion.cantidad, precio)
        self.operaciones.append({
            "activo": self.d.activos[i],
            "direccion": "largo" if posicion.direccion == LARGO else "corto",
            "fecha_entrada": self.d.fechas[posicion.idx_entrada],
            "precio_entrada": posicion.precio_entrada,
            "fecha_salida": self.d.fechas[t],
            "precio_salida": precio,
            "dias": t - posicion.idx_entrada + 1,
            "s_entrada": posicion.s_entrada,
            "regimen_entrada": posicion.regimen_entrada,
            "pnl_bruto": posicion.flujo_bruto,
            "costos": posicion.costos,
            "pnl_neto": posicion.flujo_bruto - posicion.costos,
            "motivo_salida": motivo,
        })
        self.pos[i] = None

    # ----------------------------------------------------------- paso 1: open
    def _revisar_gaps(self, t):
        """Si el open ya rebasó el SL o el TP de una posición previa, sale al open."""
        for i in range(self.n):
            posicion = self.pos[i]
            if posicion is None or posicion.idx_entrada >= t:
                continue
            precio_open = self.d.open[t, i]
            if posicion.direccion == LARGO:
                toca_sl = precio_open <= posicion.nivel_sl
                toca_tp = precio_open >= posicion.nivel_tp
            else:
                toca_sl = precio_open >= posicion.nivel_sl
                toca_tp = precio_open <= posicion.nivel_tp
            if toca_sl:
                self._cerrar(i, t, precio_open, "SL")
            elif toca_tp:
                self._cerrar(i, t, precio_open, "TP")

    def _ejecutar_ordenes(self, t):
        """Ejecuta al open de t las órdenes decididas al cierre de t−1."""
        ordenes, self.pendientes = self.pendientes, {}
        precio_open = self.d.open[t]

        # a) salidas (incluye el cierre de una reversa)
        for i, orden in ordenes.items():
            if orden.motivo_salida is not None and self.pos[i] is not None:
                self._cerrar(i, t, precio_open[i], orden.motivo_salida)

        # b) cantidades objetivo con el equity al open
        equity_open = self.cash + self._valor_posiciones(precio_open)
        cantidad_objetivo = np.zeros(self.n)
        for i in range(self.n):
            if self.pos[i] is not None:
                cantidad_objetivo[i] = self.pos[i].cantidad
        for i, orden in ordenes.items():
            abre = orden.entrada is not None and self.pos[i] is None
            ajusta = orden.entrada is None and orden.motivo_salida is None and self.pos[i] is not None
            if abre or ajusta:
                cantidad_objetivo[i] = orden.peso * equity_open / precio_open[i]

        # c) sin apalancamiento: Σ|q·P| ≤ Equity/(1 + c)
        tasa_base = self.c.comision + self.c.spread_bps / 2e4 + self.c.slippage_bps / 1e4
        bruto = np.sum(np.abs(cantidad_objetivo * precio_open))
        limite = equity_open / (1.0 + tasa_base)
        if bruto > limite:
            cantidad_objetivo *= limite / bruto

        # d) entradas y ajustes de tamaño
        for i in range(self.n):
            orden = ordenes.get(i)
            if self.pos[i] is None:
                if orden is not None and orden.entrada is not None and cantidad_objetivo[i] != 0:
                    self._abrir(i, t, cantidad_objetivo[i], orden.entrada)
            else:
                dq = cantidad_objetivo[i] - self.pos[i].cantidad
                if abs(dq) > 1e-9:
                    self._operar(i, t, dq, precio_open[i])

    # ---------------------------------------------------- paso 2: durante t
    def _revisar_sl_tp(self, t):
        """SL/TP con high/low (incluye la barra de entrada). SL primero."""
        for i in range(self.n):
            posicion = self.pos[i]
            if posicion is None:
                continue
            alto, bajo = self.d.high[t, i], self.d.low[t, i]
            if posicion.direccion == LARGO:
                toca_sl = bajo <= posicion.nivel_sl
                toca_tp = alto >= posicion.nivel_tp
            else:
                toca_sl = alto >= posicion.nivel_sl
                toca_tp = bajo <= posicion.nivel_tp
            if toca_sl:                       # convención conservadora
                self._cerrar(i, t, posicion.nivel_sl, "SL")
            elif toca_tp:
                self._cerrar(i, t, posicion.nivel_tp, "TP")

    # ------------------------------------------------------ paso 3: cierre
    def _valor_posiciones(self, precios):
        valor = 0.0
        for i in range(self.n):
            if self.pos[i] is not None:
                valor += self.pos[i].cantidad * precios[i]
        return valor

    def _cobrar_borrow_fee(self, t):
        if self.c.borrow_fee_anual == 0:
            return
        for i in range(self.n):
            posicion = self.pos[i]
            if posicion is not None and posicion.cantidad < 0:
                fee = abs(posicion.cantidad) * self.d.close[t, i] * self.c.borrow_fee_anual / config.DIAS_ANIO
                self.cash -= fee
                self.costo_dia += fee
                posicion.costos += fee

    def _regimen(self, t):
        if self.d.regimen is None:
            return -1
        return int(self.d.regimen[t])

    def _decidir_ordenes(self, t, equity):
        """Con datos ≤ t decide las órdenes para el open de t+1."""
        regimen = self._regimen(t)
        regimen_previo = self._regimen(t - 1) if t > 0 else regimen
        tabla_m = config.M_REGIMEN if self.d.m_regimen is None else self.d.m_regimen
        m = tabla_m.get(regimen, 1.0)
        en_crisis = regimen == config.CRISIS
        crisis_nueva = en_crisis and regimen_previo != config.CRISIS
        w_rp = np.ones(self.n) if self.d.w_rp is None else self.d.w_rp[t]
        corr = None if self.d.corr is None else self.d.corr[t]

        if regimen != regimen_previo:
            abiertas = sum(p is not None for p in self.pos)
            self.eventos_regimen.append({"fecha": self.d.fechas[t], "de": regimen_previo,
                                         "a": regimen, "posiciones_abiertas": abiertas})

        ordenes = {}
        s_vigente = np.zeros(self.n)     # s efectiva de lo que seguirá abierto
        candidatos = {}
        for i in range(self.n):
            s_hoy = self.d.s[t, i]
            posicion = self.pos[i]
            puede_entrar = (s_hoy != 0
                            and (self.d.entry_mask is None or self.d.entry_mask[t, i])
                            and np.isfinite(self.d.atr[t, i]) and self.d.atr[t, i] > 0
                            and not (en_crisis and abs(s_hoy) < 1))
            if posicion is not None:
                reversa = s_hoy != 0 and np.sign(s_hoy) == -posicion.direccion
                dias = t - posicion.idx_entrada + 1
                if reversa:
                    ordenes[i] = Orden(peso=0.0, motivo_salida="señal")
                elif crisis_nueva and abs(posicion.s_entrada) < 1:
                    ordenes[i] = Orden(peso=0.0, motivo_salida="cambio de régimen")
                elif dias >= posicion.max_hold:
                    ordenes[i] = Orden(peso=0.0, motivo_salida="time-stop")
                else:
                    s_vigente[i] = posicion.s_entrada
                    continue
                if reversa and puede_entrar:
                    candidatos[i] = s_hoy
            elif puede_entrar:
                candidatos[i] = s_hoy

        # Entradas: el tamaño sale de la composición del Paso 7
        if candidatos:
            s_total = s_vigente.copy()
            for i, s_i in candidatos.items():
                s_total[i] = s_i
            w_target, conflictos = pesos_objetivo(w_rp, s_total, m, corr)
            self.conflictos += conflictos
            for i, s_i in candidatos.items():
                if w_target[i] == 0:
                    continue          # la política de conflictos anuló la entrada
                info = {"direccion": int(np.sign(s_i)), "s": s_i, "atr": self.d.atr[t, i],
                        "m_sl": self.d.m_sl[t, i], "m_tp": self.d.m_tp[t, i],
                        "max_hold": int(self.d.max_hold[t, i]), "regimen": regimen}
                orden_previa = ordenes.get(i)
                motivo = orden_previa.motivo_salida if orden_previa else None
                ordenes[i] = Orden(peso=w_target[i], motivo_salida=motivo, entrada=info)

        # Revisión de rebalanceo (calendario + banda δ); inmediata si entra Crisis
        es_revision = self.d.revision is not None and self.d.revision[t]
        if (es_revision or crisis_nueva) and np.any(s_vigente != 0):
            self._revisar_rebalanceo(t, equity, w_rp, s_vigente, m, corr,
                                     ordenes, forzar=crisis_nueva)
        self.pendientes = ordenes

    def _revisar_rebalanceo(self, t, equity, w_rp, s_vigente, m, corr, ordenes, forzar):
        """Disparador híbrido: rebalancea solo si ‖w⁻ − w^target‖₁ > δ."""
        w_target, conflictos = pesos_objetivo(w_rp, s_vigente, m, corr)
        self.conflictos += conflictos
        w_antes = np.zeros(self.n)        # pesos post-drift al cierre de t
        for i in range(self.n):
            if self.pos[i] is not None and s_vigente[i] != 0:
                w_antes[i] = self.pos[i].cantidad * self.d.close[t, i] / equity
        distancia = np.sum(np.abs(w_target - w_antes))
        if forzar or distancia > self.d.delta:
            for i in range(self.n):
                if s_vigente[i] != 0:
                    ordenes[i] = Orden(peso=w_target[i])
            self.rebalanceos.append({"fecha": self.d.fechas[t],
                                     "turnover": 0.5 * distancia,
                                     "distancia_l1": distancia})

    # ------------------------------------------------------------- loop
    def correr(self):
        T, n = self.T, self.n
        equity = np.zeros(T)
        efectivo = np.zeros(T)
        cantidades = np.zeros((T, n))
        costos = np.zeros(T)
        nocional = np.zeros(T)
        bruto_max = 0.0
        for t in range(T):
            self.costo_dia = 0.0
            self.nocional_dia = 0.0
            # 1. open
            self._revisar_gaps(t)
            if self.pendientes:
                self._ejecutar_ordenes(t)
                equity_open = self.cash + self._valor_posiciones(self.d.open[t])
                bruto_open = sum(abs(p.cantidad) * self.d.open[t, i]
                                 for i, p in enumerate(self.pos) if p is not None)
                bruto_max = max(bruto_max, bruto_open / equity_open)
            # 2. durante el día
            self._revisar_sl_tp(t)
            # 3. cierre
            self._cobrar_borrow_fee(t)
            if t == T - 1 and self.d.cerrar_al_final:
                for i in range(n):
                    if self.pos[i] is not None:
                        self._cerrar(i, t, self.d.close[t, i], "fin de muestra")
            equity[t] = self.cash + self._valor_posiciones(self.d.close[t])
            efectivo[t] = self.cash
            for i in range(n):
                if self.pos[i] is not None:
                    cantidades[t, i] = self.pos[i].cantidad
            costos[t] = self.costo_dia
            nocional[t] = self.nocional_dia
            if t < T - 1:
                self._decidir_ordenes(t, equity[t])
        return self._empaquetar(equity, efectivo, cantidades, costos, nocional, bruto_max)

    def _empaquetar(self, equity, efectivo, cantidades, costos, nocional, bruto_max):
        fechas, activos = self.d.fechas, self.d.activos
        equity_s = pd.Series(equity, index=fechas, name="equity")
        posiciones = pd.DataFrame(cantidades, index=fechas, columns=activos)
        pesos = posiciones * self.d.close / equity[:, None]
        columnas_op = ["activo", "direccion", "fecha_entrada", "precio_entrada", "fecha_salida",
                       "precio_salida", "dias", "s_entrada", "regimen_entrada", "pnl_bruto",
                       "costos", "pnl_neto", "motivo_salida"]
        return ResultadoBacktest(
            equity=equity_s,
            efectivo=pd.Series(efectivo, index=fechas, name="efectivo"),
            posiciones=posiciones,
            pesos=pesos,
            operaciones=pd.DataFrame(self.operaciones, columns=columnas_op),
            costos_diarios=pd.Series(costos, index=fechas, name="costos"),
            nocional_operado=pd.Series(nocional, index=fechas, name="nocional"),
            rebalanceos=pd.DataFrame(self.rebalanceos, columns=["fecha", "turnover", "distancia_l1"]),
            eventos_regimen=pd.DataFrame(self.eventos_regimen,
                                         columns=["fecha", "de", "a", "posiciones_abiertas"]),
            conflictos=self.conflictos,
            exposicion_bruta_max=bruto_max,
        )


def run_backtest(datos, costos=None):
    """Corre el backtest y regresa un ResultadoBacktest."""
    if costos is None:
        costos = Costos()
    return _Motor(datos, costos).correr()


def datos_un_activo(ohlcv, senales, params, entry_mask=None, cerrar_al_final=False,
                    nombre="activo"):
    """Arma DatosMotor para el backtest de un solo activo (n = 1, peso 1).

    Es el mismo motor que usa el portafolio; solo cambia que no hay régimen,
    ni pesos RP, ni rebalanceo.
    """
    T = len(ohlcv)

    def columna(x):
        return np.asarray(x, dtype=float).reshape(T, 1)

    mascara = None if entry_mask is None else np.asarray(entry_mask, dtype=bool).reshape(T, 1)
    return DatosMotor(
        fechas=ohlcv.index, activos=[nombre],
        open=columna(ohlcv["open"]), high=columna(ohlcv["high"]),
        low=columna(ohlcv["low"]), close=columna(ohlcv["close"]),
        s=columna(senales["s"].fillna(0.0)), atr=columna(senales["atr"]),
        m_sl=np.full((T, 1), params["m_sl"]), m_tp=np.full((T, 1), params["m_tp"]),
        max_hold=np.full((T, 1), params["max_hold"]),
        entry_mask=mascara, volumen=columna(ohlcv["volume"]),
        cerrar_al_final=cerrar_al_final)
