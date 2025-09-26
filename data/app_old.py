# =============================================================
# Elasticity Price Monitor — Shiny for Python (LOCAL FILES)
# Converted from Streamlit → Shiny (Python) to run locally
# -------------------------------------------------------------
# Files expected in the SAME folder as this app.py:
#   - prediction_input_series.(parquet|csv)
#   - final_pe_mt.(parquet|csv)
#   - mt_consolidated_pe.(parquet|csv)   # optional in this app
#
# Run locally:
#   1) python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
#   2) pip install shiny shinywidgets plotly pandas numpy python-dateutil
#   3) shiny run --reload app.py
# =============================================================

from __future__ import annotations
import os, math, re, pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget

# ------------------------
# THEME / COLORS
# ------------------------
COCA_RED = "#E41E26"
COCA_RED_ALT = "#FF4B57"
COCA_BLACK = "#000000"
COCA_GRAY_DARK = "#4B4B4B"
COCA_GRAY_LIGHT = "#E9ECEF"
COCA_WHITE = "#FFFFFF"
RED_REAL = "#d62728"
BLACK_REF = "#000000"

px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = [
    COCA_RED, "#1F1F1F", "#666666", COCA_RED_ALT, "#2E2E2E",
    "#8C8C8C", "#B3B3B3", "#D9D9D9"
]

# ------------------------
# HELPERS
# ------------------------

def read_table_local(base: str) -> pd.DataFrame:
    """Read Parquet or CSV from the app folder, preferring Parquet."""
    p = pathlib.Path(__file__).parent
    pq = p / f"data/{base}.parquet"
    cs = p / f"data/{base}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
        # try utf-8 then latin-1
        try:
            return pd.read_csv(cs)
        except UnicodeDecodeError:
            return pd.read_csv(cs, encoding="latin-1")
    raise FileNotFoundError(f"No se encontró {base}.parquet ni {base}.csv en {p}")


def pick_col(df: pd.DataFrame, opciones):
    for c in opciones:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan]*len(df), index=df.index)


def normalize_series_schema(df_series: pd.DataFrame) -> pd.DataFrame:
    df = df_series.copy()
    # PERIOD normalization
    if "PERIOD" in df.columns:
        s = df["PERIOD"].astype(str).str.strip()
        periods = pd.to_datetime(s, format="%Y_%m", errors="coerce")
        if periods.isna().any():
            m1 = s.str.match(r"^\d{4}-\d{1,2}$")
            periods.loc[m1] = pd.to_datetime(s[m1] + "-01", errors="coerce")
            m2 = s.str.match(r"^\d{6}$")
            periods.loc[m2] = pd.to_datetime(s[m2], format="%Y%m", errors="coerce")
            m3 = s.str.match(r"^\d{8}$")
            periods.loc[m3] = pd.to_datetime(s[m3], format="%Y%m%d", errors="coerce")
        df["PERIOD"] = periods
    else:
        y = pick_col(df,["YEAR", "year"])
        m = pick_col(df,["MONTH", "month"])
        y_str = y.astype("Int64").astype(str).str.zfill(4)
        m_str = m.astype("Int64").astype(str).str.zfill(2)
        df["PERIOD"] = pd.to_datetime(y_str + "-" + m_str + "-01", errors="coerce")

    rename_map = {}
    if "CATEGORY_REVISED"    in df.columns: rename_map["CATEGORY_REVISED"]  = "CATEGORIA"
    if "SUBCATEGORY_REVISED" in df.columns: rename_map["SUBCATEGORY_REVISED"] = "SUBCATEGORIA"
    if "MARCA_REVISED_UPD"   in df.columns: rename_map["MARCA_REVISED_UPD"] = "MARCA"
    if "SKU_GROUP"           in df.columns: rename_map["SKU_GROUP"]         = "SKU"
    if "UC_CALCULATED"       in df.columns: rename_map["UC_CALCULATED"]     = "CU"
    if "UC"                  in df.columns: rename_map["UC"]                = "CU"
    if "INGRESO_NETO" in df.columns or "INGRES_NETO" in df.columns:
        rename_map["INGRESO_NETO" if "INGRESO_NETO" in df.columns else "INGRES_NETO"] = "INGRESO"
    if "PRICE"               in df.columns: rename_map["PRICE"] = "PRECIO"
    if rename_map:
        df = df.rename(columns=rename_map)

    for need in ["PERIOD", "SKU"]:
        if need not in df.columns:
            raise KeyError(f"Falta la columna requerida: {need}")

    for col in ["CU","INGRESO","PRECIO"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

    return df


def load_joined_for_elasticity(df_base: pd.DataFrame, pe_table: pd.DataFrame) -> pd.DataFrame:
    dfb = df_base.copy()
    dfb = dfb[pd.to_datetime(dfb["PERIOD"]).ge(pd.Timestamp("2021-01-01"))]

    pe = pe_table.copy()
    pe = pe.rename(columns={
        "SKU_NAME": "SKU",
        "FINAL_PE": "FINAL_PE",
        "PE_RANGE_CHECK_OVERALL": "PE_RANGE_CHECK_OVERALL",
        "P1_MIN": "P1_MIN", "P1_MAX": "P1_MAX",
        "P2_MIN": "P2_MIN", "P2_MAX": "P2_MAX",
        "FINAL_PE_SOURCE": "FINAL_PE_SOURCE",
    })

    dj = dfb.merge(pe, on="SKU", how="inner").sort_values(["SKU", "PERIOD"]).copy()
    if "NET_PE" in dj.columns:
        dj["FINAL_PE"] = pd.to_numeric(dj["NET_PE"], errors="coerce")

    price_col = next((c for c in ["PRECIO", "PRICE"] if c in dj.columns), None)
    cu_col    = "CU" if "CU" in dj.columns else None
    missing = []
    if price_col is None: missing.append("PRECIO/PRICE")
    if cu_col is None:    missing.append("CU")
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {', '.join(missing)}")

    dj[price_col] = pd.to_numeric(dj[price_col], errors="coerce")
    dj[cu_col]    = pd.to_numeric(dj[cu_col],    errors="coerce")

    dj["lag_price"] = dj.groupby("SKU")[price_col].shift(1)
    dj["lag_cu"]    = dj.groupby("SKU")[cu_col].shift(1)

    dj["dln_precio"] = np.nan
    ok_p = (dj[price_col] > 0) & (dj["lag_price"] > 0)
    dj.loc[ok_p, "dln_precio"] = np.log(dj.loc[ok_p, price_col]) - np.log(dj.loc[ok_p, "lag_price"])

    dj["dln_cu"] = np.nan
    ok_u = (dj[cu_col] > 0) & (dj["lag_cu"] > 0)
    dj.loc[ok_u, "dln_cu"] = np.log(dj.loc[ok_u, cu_col]) - np.log(dj.loc[ok_u, "lag_cu"])

    dj["FINAL_PE"] = pd.to_numeric(dj.get("FINAL_PE"), errors="coerce")
    dj["impacto_esperado"] = dj["FINAL_PE"] * dj["dln_precio"]

    return dj


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (np.abs(y_true) > eps)
    if mask.sum()==0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum()==0: return np.nan
    num = np.abs(y_true[mask]-y_pred[mask])
    den = (np.abs(y_true[mask]) + np.abs(y_pred[mask]) + eps)
    return np.mean(2*num/den) * 100


def diracc(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum()==0: return np.nan
    return (np.sign(y_true[mask])==np.sign(y_pred[mask])).mean() * 100


def safe_lndiff(series):
    s = pd.to_numeric(series, errors="coerce"); s = s.where(s > 0, np.nan)
    return np.log(s).diff()


def wavg_safe(values, weights, fallback="mean"):
    v = pd.to_numeric(values, errors="coerce"); w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        if fallback == "mean":
            return float(v.dropna().mean()) if v.notna().any() else np.nan
        return np.nan
    sw = w[mask].sum()
    if sw <= 0:
        if fallback == "mean":
            return float(v[mask].mean()) if mask.any() else np.nan
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def detect_price_col(df_):
    if "PRECIO" in df_.columns: return "PRECIO", None
    if "PRICE"  in df_.columns: return "PRICE",  None
    if ("INGRESO" in df_.columns) and ("CU" in df_.columns):
        s = np.where(
            pd.to_numeric(df_["CU"], errors="coerce") > 0,
            pd.to_numeric(df_["INGRESO"], errors="coerce") / pd.to_numeric(df_["CU"], errors="coerce"),
            np.nan
        )
        return "__PRICE_FALLBACK__", pd.Series(s, index=df_.index)
    return None, None


def apply_ranges_block(dd: pd.DataFrame) -> pd.DataFrame:
    d = dd.copy()
    for c in ["P1_MIN", "P1_MAX", "P2_MIN", "P2_MAX", "PE_RANGE_CHECK_OVERALL"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    cond1 = (d.get("PE_RANGE_CHECK_OVERALL") == 1)
    cond2 = (d.get("PE_RANGE_CHECK_OVERALL") == 2)

    e_min = np.where(cond1, d.get("P1_MIN"), np.where(cond2, d.get("P2_MIN"), np.nan))
    e_max = np.where(cond1, d.get("P1_MAX"), np.where(cond2, d.get("P2_MAX"), np.nan))

    eps_local = 5e-4
    dln_p = pd.to_numeric(d.get("dln_precio"), errors="coerce")
    dln_q = pd.to_numeric(d.get("dln_cu"), errors="coerce")
    eps_teor = (dln_q / dln_p).replace([np.inf, -np.inf], np.nan)

    no_move = dln_p.abs() < eps_local
    has_bounds = (~np.isnan(e_min)) & (~np.isnan(e_max))
    lower = np.where(has_bounds, e_min, np.nan)
    upper = np.where(has_bounds, e_max, np.nan)

    eps_used = eps_teor.copy()
    m = has_bounds & (~no_move) & eps_teor.notna()
    eps_used[m] = np.clip(eps_teor[m], lower[m], upper[m])
    eps_used[~has_bounds] = np.nan
    eps_used[no_move] = np.nan

    e_used_fill = eps_used.copy()
    e_used_fill[no_move] = 0.0
    d["real_clipped"] = e_used_fill * dln_p

    d["clip_source"] = None
    to_min = m & (eps_teor < lower)
    to_max = m & (eps_teor > upper)
    cond1_np = np.asarray(getattr(cond1, "fillna", lambda *_: cond1)(False))
    cond2_np = np.asarray(getattr(cond2, "fillna", lambda *_: cond2)(False))
    d.loc[to_min & cond1_np, "clip_source"] = "P1_MIN"
    d.loc[to_max & cond1_np, "clip_source"] = "P1_MAX"
    d.loc[to_min & cond2_np, "clip_source"] = "P2_MIN"
    d.loc[to_max & cond2_np, "clip_source"] = "P2_MAX"
    d.loc[no_move, "clip_source"] = "NO_PRICE_MOVE"

    d["epsilon_teor"] = eps_teor
    d["epsilon_used"] = eps_used
    d["elasticidad_usada_real"] = e_used_fill
    return d


# =====================
# REACTIVE DATA
# =====================

@reactive.Calc
def df_series() -> pd.DataFrame:
    return normalize_series_schema(read_table_local("prediction_input_series"))

@reactive.Calc
def df_elast() -> pd.DataFrame:
    try:
        return read_table_local("final_pe_mt")
    except FileNotFoundError:
        # Allow running without the ranges table (some tabs will degrade)
        return pd.DataFrame({"SKU": [], "FINAL_PE": []})

@reactive.Calc
def df_base():
    d = df_series().copy()
    d = d[d["PERIOD"] >= pd.Timestamp("2021-01-01")]
    return d

@reactive.Calc
def df_joined():
    base = df_base().copy()
    pe = df_elast().copy()
    if pe.empty:
        return base.assign(FINAL_PE=np.nan, PE_RANGE_CHECK_OVERALL=np.nan,
                           P1_MIN=np.nan, P1_MAX=np.nan, P2_MIN=np.nan, P2_MAX=np.nan)
    return load_joined_for_elasticity(base, pe)


# =====================
# UI
# =====================

def kpi_card(label: str, value: str):
    return ui.div(
        ui.div(label, style="font-size:12px; opacity:.85; margin-bottom:4px;"),
        ui.div(value, style="font-size:22px; font-weight:700; line-height:1;"),
        style=f"background:{COCA_WHITE}; color:{COCA_BLACK}; padding:10px 12px; border-radius:12px; border:2px solid {COCA_RED}; box-shadow:0 2px 8px rgba(0,0,0,.06); height:100%;"
    )

app_ui = ui.page_fluid(
    ui.tags.link(
        rel="stylesheet",
        href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap"
    ),
    ui.tags.style("""
        html, body { font-family: 'Montserrat', sans-serif; }
        .muted { color: #666; font-size: 12px; }
    """),

    ui.h2("Monitor de Elasticidad de Precio"),

    ui.page_sidebar(
        ui.sidebar(
            ui.output_ui("cat_sel_ui"),
            ui.output_ui("sub_sel_ui"),
            ui.output_ui("marca_sel_ui"),
            ui.output_ui("sku_sel_ui"),
            ui.output_ui("date_range_ui"),
            ui.input_slider("chart_h", "Altura de gráficas (px)", 280, 600, 380, step=10),
            ui.hr(),
            ui.input_radio_buttons(
                "level", "Nivel",
                {"Total": "Total", "Categoría": "Categoría", "Subcategoría": "Subcategoría", "SKU": "SKU"},
                selected="Total", inline=True
            ),
            ui.output_ui("topn_ui"),
            width=300
        ),
        ui.navset_tab(
            ui.nav_panel(
                "Línea de Tiempo de Elasticidad",
                ui.row(
                    ui.column(4, ui.output_ui("kpi_1")),
                    ui.column(4, ui.output_ui("kpi_2")),
                    ui.column(4, ui.output_ui("kpi_3")),
                ),
                ui.input_radio_buttons(
                    "modo_eps", "Serie mostrada",
                    ["Teórica", "Teórica ajustada"],
                    selected="Teórica ajustada", inline=True
                ),
                output_widget("fig_linea"),
                ui.download_button("dl_linea", "Descargar serie mostrada (CSV)")
            ),
            ui.nav_panel(
                "Desempeño de Elasticidad",
                ui.output_ui("perf_kpis"),
                output_widget("fig_perf")
            ),
            ui.nav_panel(
                "Clasificación de Elasticidad",
                output_widget("fig_mapa"),
                output_widget("fig_pie"),
                ui.download_button("dl_mapa", "Descargar mapa de elasticidad (CSV)")
            ),
            ui.nav_panel(
                "Explorador de Ventas",
                ui.input_radio_buttons(
                    "metric_mode", "Métrica / vista",
                    ["Volumen (CU)", "Ventas ($ Ingreso Neto)", "Precio vs. CU (Δ%)"],
                    selected="Volumen (CU)", inline=True
                ),
                output_widget("fig_explorer"),
                ui.download_button("dl_explorer", "Descargar CSV")
            ),
            ui.nav_panel(
                "Diagnóstico",
                ui.p("(Reservado — portar paneles extra si lo necesitas)")
            ),
            ui.nav_panel(
                "Descomposición de Ingresos",
                output_widget("fig_bennett"),
                ui.download_button("dl_bennett", "Descargar descomposición (CSV)")
            ),
        )
    )
)



# =====================
# SERVER
# =====================

def infer_level(cat_sel, sub_sel, sku_sel):
    if sku_sel != "All":
        return "SKU"
    elif sub_sel != "All":
        return "Subcategoría"
    elif cat_sel != "All":
        return "Categoría"
    else:
        return "Total"


def server(input, output, session):
    # --- Dynamic Categoría select (fixes lambda choices error) ---
    @output
    @render.ui
    def cat_sel_ui():
        d = df_base()
        cats = ["All"] + sorted(d.get("CATEGORIA", pd.Series(dtype=str)).dropna().unique().tolist())
        return ui.input_select("cat_sel", "Categoría", choices=cats, selected="All")

    @output
    @render.ui
    def date_range_ui():
        d = df_base()
        if len(d):
            start = pd.to_datetime(d["PERIOD"]).min().date()
            end = pd.to_datetime(d["PERIOD"]).max().date()
        else:
            start = end = datetime.today().date()
        return ui.input_date_range("date_range", "Rango de fechas", start=start, end=end)

    # -------- Dynamic sidebar inputs
    @output
    @render.ui
    def sub_sel_ui():
        d = df_base()
        cat_sel = input.cat_sel()
        if "SUBCATEGORIA" in d.columns:
            if cat_sel != "All":
                subs = ["All"] + sorted(d.loc[d["CATEGORIA"]==cat_sel, "SUBCATEGORIA"].dropna().unique().tolist())
            else:
                subs = ["All"] + sorted(d["SUBCATEGORIA"].dropna().unique().tolist())
            return ui.input_select("sub_sel", "Subcategoría", choices=subs)
        return ui.input_select("sub_sel", "Subcategoría", choices=["All"], selected="All")

    @output
    @render.ui
    def marca_sel_ui():
        d = df_base()
        cat_sel = input.cat_sel() or "All"
        sub_sel = input.sub_sel() or "All"
        if "MARCA" in d.columns:
            mask = pd.Series(True, index=d.index)
            if cat_sel != "All": mask &= (d["CATEGORIA"]==cat_sel)
            if (sub_sel != "All") and ("SUBCATEGORIA" in d.columns): mask &= (d["SUBCATEGORIA"]==sub_sel)
            marcas = ["All"] + sorted(d.loc[mask, "MARCA"].dropna().unique().tolist())
            return ui.input_select("marca_sel", "Marca", choices=marcas)
        return ui.input_select("marca_sel", "Marca", choices=["All"], selected="All")
        return ui.input_select("marca_sel", "Marca", choices=["All"], selected="All")

    @output
    @render.ui
    def sku_sel_ui():
        d = df_base()
        cat_sel = input.cat_sel() or "All"
        sub_sel = input.sub_sel() or "All"
        marca_sel = input.marca_sel() or "All"
        mask = pd.Series(True, index=d.index)
        if cat_sel != "All": mask &= (d["CATEGORIA"]==cat_sel)
        if (sub_sel != "All") and ("SUBCATEGORIA" in d.columns): mask &= (d["SUBCATEGORIA"]==sub_sel)
        if ("MARCA" in d.columns) and (marca_sel != "All"): mask &= (d["MARCA"]==marca_sel)
        skus = ["All"] + sorted(d.loc[mask, "SKU"].dropna().unique().tolist())
        return ui.input_select("sku_sel", "SKU", choices=skus)

    @output
    @render.ui
    def topn_ui():
        level_sel = input.level()
        if level_sel in ("Categoría","Subcategoría","SKU"):
            return ui.input_slider("topn", "Mostrar Top N (por volumen total)", 5, 30, 15)
        return ui.div()

    # ------------- Common reactive filtered data
    @reactive.Calc
    def filters():
        base = df_base()
        # Fecha inicial y final seguras aunque aún no exista el input
        try:
            dr = input.date_range()
            d_start = pd.Timestamp(dr[0])
            d_end = pd.Timestamp(dr[1])
        except Exception:
            if len(base):
                d_start = pd.to_datetime(base["PERIOD"]).min()
                d_end = pd.to_datetime(base["PERIOD"]).max()
            else:
                today = datetime.today().date()
                d_start = d_end = pd.Timestamp(today)

        return {
            "cat":   (input.cat_sel()   or "All"),
            "sub":   (input.sub_sel()   or "All"),
            "marca": (input.marca_sel() or "All"),
            "sku":   (input.sku_sel()   or "All"),
            "d_start": d_start,
            "d_end":   d_end,
        }

    @reactive.Calc
    def df_filtered():
        d = df_base().copy()
        f = filters()
        if f['cat']   != "All" and "CATEGORIA"    in d.columns: d = d[d["CATEGORIA"] == f['cat']]
        if f['sub']   != "All" and "SUBCATEGORIA" in d.columns: d = d[d["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA"        in d.columns: d = d[d["MARCA"] == f['marca']]
        if f['sku']   != "All" and "SKU"          in d.columns: d = d[d["SKU"] == f['sku']]
        d = d[(d["PERIOD"] >= f['d_start']) & (d["PERIOD"] <= f['d_end'])]
        return d

    @reactive.Calc
    def level_sel():
        return infer_level(input.cat_sel(), input.sub_sel(), input.sku_sel()) if input.level() == "Total" else input.level()

    # ================= TAB 1: Línea de tiempo
    @render.ui
    def kpi_1():
        return kpi_card("Meses con elasticidad calculada", "—")
    @render.ui
    def kpi_2():
        return kpi_card("Elasticidad del modelo (constante)", "—")
    @render.ui
    def kpi_3():
        return kpi_card("MAE(ε, ε_modelo)", "—")

    @render_widget
    def fig_linea():
        base = df_joined().copy()
        f = filters()
        if f['cat']   != "All" and "CATEGORIA"   in base.columns: base = base[base["CATEGORIA"] == f['cat']]
        if f['sub']   != "All" and "SUBCATEGORIA" in base.columns: base = base[base["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA"       in base.columns: base = base[base["MARCA"] == f['marca']]
        if f['sku']   != "All" and "SKU"        in base.columns: base = base[base["SKU"] == f['sku']]
        base = base[(base["PERIOD"] >= f['d_start']) & (base["PERIOD"] <= f['d_end'])].copy()
        if base.empty:
            return go.Figure()

        for c in ["dln_precio", "dln_cu", "FINAL_PE", "CU"]:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")
        base = base.sort_values(["SKU","PERIOD"]).copy()
        base["CU_lag"] = base.groupby("SKU", sort=False)["CU"].shift(1)
        base["PERIOD_M"] = pd.to_datetime(base["PERIOD"]).dt.to_period("M").dt.to_timestamp()

        usar_ajustada = (input.modo_eps() == "Teórica ajustada")
        if usar_ajustada:
            work = apply_ranges_block(base.copy())
            col_eps = "epsilon_used"
        else:
            if "epsilon_teor" not in base.columns:
                dln_p = pd.to_numeric(base.get("dln_precio"), errors="coerce")
                dln_q = pd.to_numeric(base.get("dln_cu"), errors="coerce")
                base["epsilon_teor"] = (dln_q / dln_p).replace([np.inf, -np.inf], np.nan)
            work = base.copy(); col_eps = "epsilon_teor"

        eps_local = 1e-8
        work["__valid"] = (
            work[col_eps].notna() &
            work["CU_lag"].notna() & (work["CU_lag"] > 0) &
            work["dln_precio"].abs().gt(eps_local)
        )

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "SKU": "SKU"}
        entity_col = entity_map.get(level_sel())

        def monthly_eps_weighted(df_in: pd.DataFrame, col_eps: str, entity_col: str | None):
            d = df_in.copy()
            eps_local = 5e-4
            mvalid = (
                d[col_eps].notna()
                & d["CU_lag"].notna() & (pd.to_numeric(d["CU_lag"], errors="coerce") > 0)
                & pd.to_numeric(d["dln_precio"], errors="coerce").abs().gt(eps_local)
                & d["PERIOD_M"].notna()
            )
            d = d[mvalid].copy()
            if d.empty:
                return pd.DataFrame({"PERIOD_M": [], "eps": []})

            if (entity_col is None) or (entity_col not in d.columns):
                tot = d.groupby("PERIOD_M", dropna=False).agg(
                    w=("CU_lag", lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1)),
                    wsum=(col_eps, lambda s: np.nansum(
                        pd.to_numeric(s, errors="coerce") * pd.to_numeric(d.loc[s.index, "CU_lag"], errors="coerce")
                    )),
                ).reset_index()
                tot["eps"] = tot["wsum"] / tot["w"].replace(0, np.nan)
                return tot[["PERIOD_M", "eps"]]

            grp = d.groupby(["PERIOD_M", entity_col], dropna=False)
            tmp = grp.agg(
                w_ent=("CU_lag", lambda s: pd.to_numeric(s, errors="coerce").sum(min_count=1)),
                wsum_ent=(col_eps, lambda s: np.nansum(
                    pd.to_numeric(s, errors="coerce") * pd.to_numeric(d.loc[s.index, "CU_lag"], errors="coerce")
                )),
            ).reset_index()
            tmp["eps_ent"] = tmp["wsum_ent"] / tmp["w_ent"].replace(0, np.nan)

            def collapse(g):
                e = pd.to_numeric(g["eps_ent"], errors="coerce")
                w = pd.to_numeric(g["w_ent"], errors="coerce")
                m = e.notna() & w.notna() & (w > 0)
                if not m.any():
                    return np.nan
                return float(np.average(e[m], weights=w[m]))

            out = (tmp.groupby("PERIOD_M", dropna=False)
                    .apply(collapse)
                    .rename("eps")
                    .reset_index())
            return out

        serie = monthly_eps_weighted(work, col_eps, entity_col)
        eps_model = wavg_safe(work["FINAL_PE"], work["CU_lag"])
        serie["eps_model"] = eps_model

        # KPIs (update text)
        session.send_custom_message("kpi1_text", dict(v=int(serie['eps'].notna().sum())))
        session.send_custom_message("kpi2_text", dict(v=(f"{eps_model:.6f}" if pd.notna(eps_model) else "-")))
        mae_vs_model = (serie["eps"] - serie["eps_model"]).abs().mean()
        session.send_custom_message("kpi3_text", dict(v=(f"{mae_vs_model:.2f}" if pd.notna(mae_vs_model) else "-")))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=serie["PERIOD_M"], y=serie["eps"],
            mode="lines+markers", name=("Elasticidad teórica ajustada" if usar_ajustada else "Elasticidad teórica"),
            line=dict(width=2, color=COCA_RED)
        ))
        fig.add_trace(go.Scatter(
            x=serie["PERIOD_M"], y=serie["eps_model"],
            mode="lines", name="Elasticidad del modelo (constante)", line=dict(width=2, dash="dash", color=BLACK_REF)
        ))
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        fig.update_layout(
            title=f"{'Elasticidad teórica ajustada' if usar_ajustada else 'Elasticidad teórica'} vs. modelo — Nivel: {level_sel()}",
            yaxis_title="Elasticidad",
            hovermode="x unified", height=int(input.chart_h()),
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        return fig

    @render.download(filename=lambda: f"linea_tiempo_elasticidad_{'ajustada' if input.modo_eps()== 'Teórica ajustada' else 'teorica'}_{level_sel().lower()}.csv")
    def dl_linea():
        base = df_joined().copy()
        if base.empty:
            yield "".encode("utf-8")
            return
        # Simple output for now (period + placeholders)
        out = base[["PERIOD"]].drop_duplicates().sort_values("PERIOD").rename(columns={"PERIOD":"PERIOD"})
        yield out.to_csv(index=False).encode("utf-8")

    # ================= TAB 2: Desempeño (simplified total/SKU)
    @output
    @render.ui
    def perf_kpis():
        return ui.row(
            ui.column(4, kpi_card("MAPE (↓ mejor)", "—")),
            ui.column(4, kpi_card("SMAPE (↓ mejor)", "—")),
            ui.column(4, kpi_card("DIR (↑ mejor)", "—")),
        )

    @render_widget
    def fig_perf():
        dj = df_joined().copy()
        f = filters()
        if f['cat']   != "All" and "CATEGORIA"    in dj.columns: dj = dj[dj["CATEGORIA"] == f['cat']]
        if f['sub']   != "All" and "SUBCATEGORIA" in dj.columns: dj = dj[dj["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA"        in dj.columns: dj = dj[dj["MARCA"] == f['marca']]
        dj = dj[(dj["PERIOD"] >= f['d_start']) & (dj["PERIOD"] <= f['d_end'])]
        if dj.empty:
            return go.Figure()
        dj = apply_ranges_block(dj)
        # Total view (expected vs real adjusted)
        grp = dj.groupby("PERIOD", dropna=False)
        y_real = grp["real_clipped"].sum(min_count=1)
        x_exp  = grp["impacto_esperado"].sum(min_count=1)
        x = pd.to_numeric(x_exp, errors="coerce"); y = pd.to_numeric(y_real, errors="coerce")
        mask = x.replace([np.inf,-np.inf], np.nan).notna() & y.replace([np.inf,-np.inf], np.nan).notna()
        x, y = x[mask], y[mask]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Períodos", marker=dict(size=7, color=RED_REAL)))
        dom = np.array([np.nanmin(np.r_[x.values, y.values]), np.nanmax(np.r_[x.values, y.values])]) if len(x) else np.array([0,1])
        fig.add_trace(go.Scatter(x=dom, y=dom, mode="lines", name="y=x", line=dict(color=BLACK_REF, width=2, dash="dash")))
        fig.update_layout(xaxis_title="Impacto esperado (Δln CU)", yaxis_title="Impacto real ajustado (Δln CU)", height=420, margin=dict(l=10,r=10,t=10,b=10))
        return fig

    # ================= TAB 3: Clasificación (simplified)
    def classify_elasticity(e):
        if pd.isna(e): return "Sin dato"
        ae = abs(float(e))
        if math.isclose(ae, 1.0, rel_tol=1e-3, abs_tol=1e-3): return "Unitario"
        return "Inelástico (<1)" if ae < 1 else "Elástico (>1)"

    @render_widget
    def fig_mapa():
        dj5 = df_joined().copy()
        f = filters()
        if f['cat']   != "All" and "CATEGORIA"    in dj5.columns: dj5 = dj5[dj5["CATEGORIA"] == f['cat']]
        if f['sub']   != "All" and "SUBCATEGORIA" in dj5.columns: dj5 = dj5[dj5["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA"        in dj5.columns: dj5 = dj5[dj5["MARCA"] == f['marca']]
        dj5 = dj5[(dj5["PERIOD"] >= f['d_start']) & (dj5["PERIOD"] <= f['d_end'])]
        if dj5.empty:
            return go.Figure()
        for c in ["CU","FINAL_PE"]:
            if c in dj5.columns: dj5[c] = pd.to_numeric(dj5[c], errors="coerce")
        dj5 = dj5.sort_values(["SKU","PERIOD"]).copy()
        dj5["CU_lag"] = dj5.groupby("SKU", sort=False)["CU"].shift(1)
        # choose entity col per level
        lvl = level_sel()
        if lvl == "Total":
            entity_col = "CATEGORIA" if "CATEGORIA" in dj5.columns else "SKU"
            nivel_txt  = "Categoría" if entity_col == "CATEGORIA" else "SKU"
        elif lvl == "Categoría":
            entity_col = "SUBCATEGORIA" if "SUBCATEGORIA" in dj5.columns else "SKU"
            nivel_txt  = "Subcategoría" if entity_col == "SUBCATEGORIA" else "SKU"
        else:
            entity_col = "SKU"; nivel_txt = "SKU"
        if entity_col not in dj5.columns:
            return go.Figure()
        agg_df = (
            dj5.groupby([entity_col], dropna=False)
               .apply(lambda g: wavg_safe(g["FINAL_PE"], g["CU_lag"]))
               .reset_index(name="Elasticidad_prom")
        )
        agg_df["Clase"] = agg_df["Elasticidad_prom"].apply(classify_elasticity)
        order_y = (
            agg_df[[entity_col, "Elasticidad_prom"]]
            .sort_values("Elasticidad_prom", ascending=False)[entity_col].astype(str).tolist()
        )
        color_map = {"Elástico (>1)": COCA_RED, "Unitario": COCA_BLACK, "Inelástico (<1)": COCA_GRAY_DARK}
        fig = go.Figure()
        for cls in ["Elástico (>1)", "Unitario", "Inelástico (<1)"]:
            sub = agg_df[agg_df["Clase"] == cls]
            if sub.empty: continue
            fig.add_trace(go.Bar(orientation="h", x=sub["Elasticidad_prom"], y=sub[entity_col].astype(str), name=cls, marker_color=color_map.get(cls, "#B3B3B3")))
        fig.update_layout(
            title=f"Elasticidad promedio por {nivel_txt}", xaxis_title="Elasticidad precio de la demanda (ε)", yaxis_title=nivel_txt,
            height=420, margin=dict(l=10, r=10, t=40, b=10), barmode="overlay",
            yaxis=dict(categoryorder="array", categoryarray=order_y), legend=dict(orientation="h")
        )
        fig.add_vline(x=1, line_width=2, line_dash="dot", line_color="#999999")
        fig.add_vline(x=-1, line_width=2, line_dash="dot", line_color="#999999")
        return fig

    @render_widget
    def fig_pie():
        dj5 = df_joined().copy()
        if dj5.empty:
            return go.Figure()
        dj5["CU_lag"] = dj5.groupby("SKU", sort=False)["CU"].shift(1)
        agg_df = (
            dj5.groupby(["SKU"], dropna=False)
               .apply(lambda g: wavg_safe(g["FINAL_PE"], g["CU_lag"]))
               .reset_index(name="Elasticidad_prom")
        )
        def classify(e):
            if pd.isna(e): return "Sin dato"
            ae = abs(float(e))
            if math.isclose(ae, 1.0, rel_tol=1e-3, abs_tol=1e-3): return "Unitario"
            return "Inelástico (<1)" if ae < 1 else "Elástico (>1)"
        agg_df["Clase"] = agg_df["Elasticidad_prom"].apply(classify)
        dist = agg_df["Clase"].value_counts(dropna=False).rename_axis("Clase").reset_index(name="count")
        color_map = {"Elástico (>1)": COCA_RED, "Unitario": COCA_BLACK, "Inelástico (<1)": COCA_GRAY_DARK, "Sin dato": "#B3B3B3"}
        fig = px.pie(dist, values="count", names="Clase", title="Distribución de clases", color="Clase", color_discrete_map=color_map)
        fig.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10))
        return fig

    @render.download(filename="mapa_elasticidad.csv")
    def dl_mapa():
        dj = df_joined().copy()
        if dj.empty:
            yield b""
            return
        yield dj.to_csv(index=False).encode("utf-8")

    # ================= TAB 4: Explorador (Total view: Precio vs CU Δ%)
    @render_widget
    def fig_explorer():
        d = df_filtered().copy()
        metric_mode = input.metric_mode()
        metric_col = "CU" if metric_mode == "Volumen (CU)" else ("INGRESO" if metric_mode == "Ventas ($ Ingreso Neto)" else "CU")
        if metric_mode == "Precio vs. CU (Δ%)":
            pcol, fallback_series = detect_price_col(d)
            if pcol is None:
                return go.Figure()
            if pcol == "__PRICE_FALLBACK__":
                d[pcol] = fallback_series
            if "INGRESO" in d.columns and "CU" in d.columns:
                dm = d.groupby(["PERIOD"], dropna=False).agg(CU=("CU","sum"), INGRESO=("INGRESO","sum")).reset_index()
                dm["PRICE_MEAN"] = np.where(dm["CU"]>0, dm["INGRESO"]/dm["CU"], np.nan)
            else:
                dm = d.groupby(["PERIOD"], dropna=False).agg(CU=("CU","sum"), PRICE_MEAN=(pcol,"mean")).reset_index()
            dm = dm.sort_values("PERIOD").copy()
            dm["dln_price"] = safe_lndiff(dm["PRICE_MEAN"])
            dm["dln_cu"]    = safe_lndiff(dm["CU"])
            dm["Δ Precio (%)"] = np.expm1(dm["dln_price"]) * 100
            dm["Δ CU (%)"]     = np.expm1(dm["dln_cu"]) * 100
            dm["PRICE_MEAN_prev"] = dm["PRICE_MEAN"].shift(1)
            dm["CU_prev"]         = dm["CU"].shift(1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dm["PERIOD"], y=dm["Δ Precio (%)"], mode="lines", name="Δ Precio (%)", line=dict(width=2, color=COCA_RED),
                customdata=np.stack([dm["PRICE_MEAN"], dm["PRICE_MEAN_prev"]], axis=-1),
                hovertemplate=("%{x|%Y-%m}<br>Δ Precio: %{y:.1f}%<br>Precio actual: %{customdata[0]:,.2f}<br>Precio previo: %{customdata[1]:,.2f}<extra></extra>")
            ))
            fig.add_trace(go.Scatter(
                x=dm["PERIOD"], y=dm["Δ CU (%)"], mode="lines", name="Δ CU (%)", line=dict(width=2, color=BLACK_REF),
                customdata=np.stack([dm["CU"], dm["CU_prev"]], axis=-1),
                hovertemplate=("%{x|%Y-%m}<br>Δ CU: %{y:.1f}%<br>CU actual: %{customdata[0]:,.0f}<br>CU previo: %{customdata[1]:,.0f}<extra></extra>")
            ))
            fig.update_layout(title="Δ% mensual: Precio vs. CU — Total", hovermode="x unified", showlegend=True,
                              yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT, title="Δ% vs. mes anterior"),
                              xaxis=dict(showgrid=False), height=int(input.chart_h()), margin=dict(l=10,r=10,t=30,b=10))
            fig.add_hline(y=0, line_width=1, line_dash="dot")
            return fig
        else:
            group = ["PERIOD"]; entity_col = None
            lvl = level_sel()
            if lvl=="Categoría" and "CATEGORIA" in d.columns:
                group.append("CATEGORIA"); entity_col = "CATEGORIA"
            elif lvl=="Subcategoría" and "SUBCATEGORIA" in d.columns:
                group.append("SUBCATEGORIA"); entity_col = "SUBCATEGORIA"
            elif lvl=="SKU" and "SKU" in d.columns:
                group.append("SKU"); entity_col = "SKU"
            agg = d.groupby(group, dropna=False)[metric_col].sum().reset_index()
            titulo_metric = "CU" if metric_col == "CU" else "Ingreso Neto"
            fig = px.line(agg, x="PERIOD", y=metric_col, color=(entity_col if entity_col in (agg.columns) else None), title=f"{lvl} mensual — {titulo_metric}")
            fig.update_traces(mode="lines", line=dict(width=2), hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m}: %{y:,}")
            fig.update_layout(title=dict(x=0.01), hovermode="x unified", showlegend=False, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT), height=int(input.chart_h()), margin=dict(l=10,r=10,t=30,b=10))
            return fig

    @render.download(filename=lambda: f"{level_sel()}_{'cu' if input.metric_mode()=='Volumen (CU)' else 'ingreso_neto'}.csv")
    def dl_explorer():
        d = df_filtered().copy()
        yield d.to_csv(index=False).encode("utf-8")

    # ================= TAB 6: Bennett (simplified)
    @render_widget
    def fig_bennett():
        d_rev = df_filtered().copy()
        if not {"INGRESO","CU"}.issubset(set(d_rev.columns)):
            return go.Figure()
        d_rev = d_rev.sort_values(["SKU","PERIOD"]).copy()
        d_rev["P"] = np.where(pd.to_numeric(d_rev["CU"], errors="coerce")>0, pd.to_numeric(d_rev["INGRESO"], errors="coerce")/pd.to_numeric(d_rev["CU"], errors="coerce"), np.nan)
        d_rev["P_lag"] = d_rev.groupby("SKU")["P"].shift(1)
        d_rev["Q_lag"] = d_rev.groupby("SKU")["CU"].shift(1)
        d_rev["P_diff"] = d_rev["P"] - d_rev["P_lag"]
        d_rev["Q_diff"] = d_rev["CU"] - d_rev["Q_lag"]
        df_ben = (d_rev.assign(price_eff = d_rev["Q_lag"] * d_rev["P_diff"], vol_eff = d_rev["P_lag"] * d_rev["Q_diff"], mix_eff = d_rev["P_diff"] * d_rev["Q_diff"]).groupby("PERIOD", dropna=False)[["price_eff","vol_eff","mix_eff"]].sum(min_count=1).reset_index().sort_values("PERIOD"))
        df_long = df_ben.melt(id_vars="PERIOD", var_name="Componente", value_name="ΔIngresos")
        names_map = {"price_eff":"Precio", "vol_eff":"Volumen", "mix_eff":"Mix"}
        df_long["Componente"] = df_long["Componente"].map(names_map)
        fig = px.bar(df_long, x="PERIOD", y="ΔIngresos", color="Componente", barmode="relative", title="ΔIngresos por componente (Bennett)")
        fig.update_layout(height=440, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
        fig.add_hline(y=0, line_width=3, line_dash="solid", line_color="black")
        return fig

    @render.download(filename="descomposicion_ingresos_bennett.csv")
    def dl_bennett():
        d = df_filtered().copy()
        yield d.to_csv(index=False).encode("utf-8")


app = App(app_ui, server)
