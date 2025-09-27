from __future__ import annotations
import os, math, pathlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots


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

CHART_H = 380  # altura estándar para todas las gráficas

px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = [
    COCA_RED, "#1F1F1F", "#666666", COCA_RED_ALT, "#2E2E2E",
    "#8C8C8C", "#B3B3B3", "#D9D9D9"
]

# ------------------------
# HELPERS
# ------------------------
def _norm_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    try:
        s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    except Exception:
        pass
    return s.str.upper()

def _norm_val(x) -> str:
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return "ALL"
    return _norm_series(pd.Series([x])).iloc[0]

# --- Qué columnas mostrar de KPIs según los filtros actuales
def _kpi_show_flags(f: dict) -> dict:
    """Devuelve flags para mostrar columnas: cat/sub/marca/sku."""
    cat  = (f.get("cat")   or "All")
    sub  = (f.get("sub")   or "All")
    mar  = (f.get("marca") or "All")
    sku  = (f.get("sku")   or "All")

    if sku != "All":
        return {"cat": True, "sub": True, "marca": True, "sku": True}
    if mar != "All":
        return {"cat": True, "sub": True, "marca": True, "sku": False}
    if sub != "All":
        return {"cat": True, "sub": True, "marca": False, "sku": False}
    # sólo categoría (si es All = total ponderado; si se elige 1 categoría, muestra su agregado)
    return {"cat": True, "sub": False, "marca": False, "sku": False}

def read_table_local(base: str) -> pd.DataFrame:
    """Read Parquet or CSV from ./data, preferring Parquet."""
    p = pathlib.Path(__file__).parent / "data"
    pq = p / f"{base}.parquet"
    cs = p / f"{base}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if cs.exists():
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
            m1 = s.str.match(r"^\d{4}-\d{1,2}$"); periods.loc[m1] = pd.to_datetime(s[m1] + "-01", errors="coerce")
            m2 = s.str.match(r"^\d{6}$");         periods.loc[m2] = pd.to_datetime(s[m2], format="%Y%m", errors="coerce")
            m3 = s.str.match(r"^\d{8}$");         periods.loc[m3] = pd.to_datetime(s[m3], format="%Y%m%d", errors="coerce")
        df["PERIOD"] = periods
    else:
        y = pick_col(df,["YEAR", "year"]); m = pick_col(df,["MONTH", "month"])
        y_str = y.astype("Int64").astype(str).str.zfill(4)
        m_str = m.astype("Int64").astype(str).str.zfill(2)
        df["PERIOD"] = pd.to_datetime(df["PERIOD"], errors="coerce")

    # rename to normalized names
    rename_map = {}
    if "CATEGORY_REVISED"    in df.columns: rename_map["CATEGORY_REVISED"]    = "CATEGORIA"
    if "SUBCATEGORY_REVISED" in df.columns: rename_map["SUBCATEGORY_REVISED"] = "SUBCATEGORIA"
    if "MARCA_REVISED_UPD"   in df.columns: rename_map["MARCA_REVISED_UPD"]   = "MARCA"
    if "SKU_GROUP"           in df.columns: rename_map["SKU_GROUP"]           = "SKU"
    if "UC_CALCULATED"       in df.columns: rename_map["UC_CALCULATED"]       = "CU"
    if "UC"                  in df.columns: rename_map["UC"]                  = "CU"
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

    pe = pe_table.copy().rename(columns={
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
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum()==0: return np.nan
    num = np.abs(y_true[mask]-y_pred[mask]); den = (np.abs(y_true[mask]) + np.abs(y_pred[mask]) + eps)
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
        return float(v.dropna().mean()) if (fallback=="mean" and v.notna().any()) else np.nan
    sw = w[mask].sum()
    if sw <= 0:
        return float(v[mask].mean()) if (fallback=="mean" and mask.any()) else np.nan
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
        return base.assign(
            FINAL_PE=np.nan, PE_RANGE_CHECK_OVERALL=np.nan,
            P1_MIN=np.nan, P1_MAX=np.nan, P2_MIN=np.nan, P2_MAX=np.nan
        )
    return load_joined_for_elasticity(base, pe)

# ---- mt_consolidated_pe
@reactive.Calc
def df_mt_pe() -> pd.DataFrame:
    try:
        d = read_table_local("mt_consolidated_pe").copy()
    except FileNotFoundError:
        return pd.DataFrame(columns=["SKU","CATEGORIA","SUBCATEGORIA","MARCA","PPG_PE","VTM_RATIO","NET_PE"])
    rename_map = {
        "PPG": "SKU",
        "CATEGORY": "CATEGORIA",
        "SUB_CATEGORY": "SUBCATEGORIA",
        "MARCA_REVISED_UPD": "MARCA"
    }
    d = d.rename(columns=rename_map)
    keep = ["Channel","SKU","CATEGORIA","SUBCATEGORIA","MARCA","PPG_PE","VTM_RATIO","NET_PE"]
    d = d[[c for c in keep if c in d.columns]].copy()
    if "Channel" in d.columns:
        d = d[(d["Channel"].fillna("").str.upper()=="MT") | (d["Channel"].isna())].copy()
    for c in ["PPG_PE","VTM_RATIO","NET_PE"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d

@reactive.Calc
def df_vtm_map() -> pd.DataFrame:
    try:
        d = read_table_local("udm_vtm_melted_mapping")  # parquet/csv en ./data
    except FileNotFoundError:
        p = pathlib.Path(__file__).parent / "data" / "udm_vtm_melted_mapping.csv"
        if p.exists():
            d = pd.read_csv(p)
        else:
            return pd.DataFrame(columns=[
                "SOURCE_SKU","DESTINATION_SKU","SRC_MFG","DEST_MFG",
                "VOLUME_TRANSFER","UNIT_MIX","REVISED_CATEGORY","MARCA_REVISED_V2"
            ])

    # columnas esperadas
    rename = {
        "SOURCE_SKU":"SOURCE_SKU",
        "DESTINATION_SKU":"DESTINATION_SKU",
        "SRC_MFG":"SRC_MFG",
        "DEST_MFG":"DEST_MFG",
        "VOLUME_TRANSFER":"VOLUME_TRANSFER",
        "UNIT_MIX":"UNIT_MIX",
        "REVISED_CATEGORY":"REVISED_CATEGORY",
        "MARCA_REVISED_V2":"MARCA_REVISED_V2",
    }
    d = d.rename(columns=rename)

    for c in ["VOLUME_TRANSFER","UNIT_MIX"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # llaves normalizadas
    d["SOURCE_SKU_KEY"] = _norm_series(d["SOURCE_SKU"])
    d["DEST_SKU_KEY"]   = _norm_series(d["DESTINATION_SKU"])
    d["SRC_MFG_KEY"]    = _norm_series(d.get("SRC_MFG", pd.Series(index=d.index, dtype=str)))
    d["DEST_MFG_KEY"]   = _norm_series(d.get("DEST_MFG", pd.Series(index=d.index, dtype=str)))
    d["IS_SRC_COCA"]    = d["SRC_MFG_KEY"].str.contains("COCA", na=False)
    d["IS_DEST_COCA"]   = d["DEST_MFG_KEY"].str.contains("COCA", na=False)
    return d


def last_full_year_window(dseries: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (start, end_exclusive) del último año completo según PERIOD mensual disponible."""
    if dseries.empty:
        today = pd.Timestamp.today().normalize()
        end_m = pd.Timestamp(year=today.year, month=today.month, day=1)
    else:
        end_m = pd.to_datetime(dseries["PERIOD"]).max()
        end_m = pd.Timestamp(year=end_m.year, month=end_m.month, day=1)
    start_m = (end_m - pd.offsets.DateOffset(months=11)).normalize()
    return (start_m, (end_m + pd.offsets.MonthEnd(1)))

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

        /* ---- TABS pro: 1 fila, centrados, 2 líneas, sin scroll ---- */
        .nav.nav-tabs {
          display: flex;
          flex-wrap: nowrap;
          border-bottom: 2px solid #E9ECEF;
          gap: 4px;
        }
        .nav.nav-tabs .nav-item {
          flex: 1 1 0;
          min-width: 0;
        }
        .nav.nav-tabs .nav-link {
          width: 100%;
          text-align: center;
          font-weight: 600;
          font-size: 13px;
          line-height: 1.15;
          white-space: normal;
          height: 2.7em;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #4B4B4B;
          padding: 8px 6px;
        }
        .nav.nav-tabs .nav-link.active {
          color: #E41E26 !important;
          border-color: #E41E26 #E41E26 #fff !important;
        }
    """),

    ui.h2("Monitor de Elasticidad de Precio"),

    ui.page_sidebar(
        ui.sidebar(
            ui.output_ui("cat_sel_ui"),
            ui.output_ui("sub_sel_ui"),
            ui.output_ui("marca_sel_ui"),
            ui.output_ui("sku_sel_ui"),
            ui.output_ui("date_range_ui"),
            width=200
        ),

        ui.navset_tab(
            ui.nav_panel(
                ui.tags.span("Elasticidad vs", ui.br(), "VTM"),
                ui.div(
                    ui.row(
                        ui.column(12, output_widget("fig_ev_correlation_timeline")),
                    ),
                    style="margin-top:16px; margin-bottom:10px;"
                ),
                ui.div(
                    ui.row(
                        ui.column(6,  output_widget("fig_ev_scatter_vtm_gap")),
                        ui.column(6,  output_widget("fig_ev_stack_flows")),
                    ),
                    style="margin-bottom:10px;"
                ),
                ui.download_button("dl_ev_monthly_flows", "Descargar (flows mensuales) CSV")
            ),

            # ---- NUEVA pestaña 1
            ui.nav_panel(
                ui.tags.span("Elasticidad &", ui.br(), "VTM Insights"),
                # FILA 1: PPG_PE
                ui.div(
                    ui.row(
                        ui.column(3, ui.output_ui("kpi_ppg_cat")),
                        ui.column(3, ui.output_ui("kpi_ppg_sub")),
                        ui.column(3, ui.output_ui("kpi_ppg_marca")),
                        ui.column(3, ui.output_ui("kpi_ppg_sku")),
                    ),
                    style="margin-top:16px; margin-bottom:10px;"
                ),
                # FILA 2: VTM_RATIO
                ui.div(
                    ui.row(
                        ui.column(3, ui.output_ui("kpi_vtm_cat")),
                        ui.column(3, ui.output_ui("kpi_vtm_sub")),
                        ui.column(3, ui.output_ui("kpi_vtm_marca")),
                        ui.column(3, ui.output_ui("kpi_vtm_sku")),
                    ),
                    style="margin-bottom:10px;"
                ),
                # FILA 3: NET_PE
                ui.div(
                    ui.row(
                        ui.column(3, ui.output_ui("kpi_net_cat")),
                        ui.column(3, ui.output_ui("kpi_net_sub")),
                        ui.column(3, ui.output_ui("kpi_net_marca")),
                        ui.column(3, ui.output_ui("kpi_net_sku")),
                    ),
                    style="margin-bottom:10px;"
                ),
                output_widget("fig_transfer_sankey"),
                output_widget("fig_transfer_topdest"),
                ui.download_button("dl_vtm", "Descargar métricas (CSV)")
            ),
            # ---- Línea de tiempo (modificada)
            ui.nav_panel(
                ui.tags.span("Línea de tiempo", ui.br(), "de Elasticidad"),
                ui.div(
                    ui.row(
                        ui.column(4, ui.output_ui("kpi_1")),
                        ui.column(4, ui.output_ui("kpi_2")),
                        ui.column(4, ui.output_ui("kpi_3")),
                    ),
                    style="margin-top:16px; margin-bottom:10px;"
                ),
                ui.input_radio_buttons(
                    "modo_eps", "Serie mostrada",
                    ["Teórica", "Teórica Ajustada"],
                    selected="Teórica Ajustada", inline=True
                ),
                output_widget("fig_linea"),
                ui.download_button("dl_linea", "Descargar serie mostrada (CSV)")
            ),
            ui.nav_panel(
                ui.tags.span("Clasificación", ui.br(), "de Elasticidad"),
                output_widget("fig_mapa"),
                output_widget("fig_pie"),
                ui.download_button("dl_mapa", "Descargar mapa de elasticidad (CSV)")
            ),
            ui.nav_panel(
                ui.tags.span("Explorador de", ui.br(), "Ventas"),
                ui.input_radio_buttons(
                    "metric_mode", "Métrica / vista",
                    ["Volumen (CU)", "Ventas ($ Ingreso Neto)", "Precio vs. CU (Δ%)"],
                    selected="Volumen (CU)", inline=True
                ),
                output_widget("fig_explorer"),
                ui.download_button("dl_explorer", "Descargar CSV")
            ),
            ui.nav_panel(
                ui.tags.span("Descomposición de", ui.br(), "Ingresos"),
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
    # --- Dynamic inputs ---
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

    # ------------- Common reactive filtered data
    @reactive.Calc
    def filters():
        base = df_base()
        try:
            dr = input.date_range()
            d_start = pd.Timestamp(dr[0]); d_end = pd.Timestamp(dr[1])
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
        return infer_level(input.cat_sel() or "All", input.sub_sel() or "All", input.sku_sel() or "All")
    
    def _flows_mask_for_filters(flows: pd.DataFrame, f: dict) -> pd.Series:
        m = pd.Series(True, index=flows.index)
        cat_key   = _norm_val(f.get("cat") if f.get("cat") != "All" else "")
        sub_key   = _norm_val(f.get("sub") if f.get("sub") != "All" else "")
        brand_key = _norm_val(f.get("marca") if f.get("marca") != "All" else "")
        sku_key   = _norm_val(f.get("sku") if f.get("sku") != "All" else "")

        if cat_key != "ALL":   m &= (flows["SRC_CAT_KEY"]  == cat_key)
        if sub_key != "ALL":   m &= (flows["SRC_SUB_KEY"]  == sub_key)
        if brand_key != "ALL": m &= (flows["SRC_BRAND_KEY"]== brand_key)
        if sku_key != "ALL":   m &= (_norm_series(flows["SRC_SKU"]) == sku_key)
        return m


    # -------- VTM context (último año)
    @reactive.Calc
    def vtm_context():
        d_base = df_base().copy()
        win_start, win_end = last_full_year_window(d_base)
        d_last = d_base[(d_base["PERIOD"] >= win_start) & (d_base["PERIOD"] < win_end)].copy()

        w = (d_last.groupby("SKU", dropna=False)["CU"]
            .sum(min_count=1)
            .rename("W_CU")
            .reset_index())

        mt = df_mt_pe().copy()
        if mt.empty:
            return {"last": d_last, "weights": w, "joined": pd.DataFrame()}

        mt2 = mt.dropna(subset=["SKU"]).copy()
        j = mt2.merge(w, on="SKU", how="left")
        j["W_CU"] = pd.to_numeric(j["W_CU"], errors="coerce").fillna(0.0)

        # ===== Normaliza claves para comparaciones robustas
        for col in ["CATEGORIA", "SUBCATEGORIA", "MARCA", "SKU"]:
            if col in j.columns:
                j[col + "_KEY"] = _norm_series(j[col])
        return {"last": d_last, "weights": w, "joined": j}


    def _wavg(df, col):
        v = pd.to_numeric(df[col], errors="coerce")
        w = pd.to_numeric(df["W_CU"], errors="coerce").clip(lower=0)
        m = v.notna() & w.notna()
        if not m.any() or w[m].sum() == 0:
            return np.nan
        return float(np.average(v[m], weights=w[m]))
    
    # def _wavg_by_level(df, metric, level_col):
    #     """Promedio ponderado por CU sobre el filtro actual, mostrando etiqueta por nivel."""
    #     if df.empty or metric not in df.columns:
    #         return np.nan
    #     v = pd.to_numeric(df[metric], errors="coerce")
    #     w = pd.to_numeric(df.get("W_CU"), errors="coerce")
    #     m = v.notna() & w.notna() & (w > 0)
    #     if not m.any():
    #         return np.nan
    #     # Ponderado sobre todas las filas; el nivel solo es para el label/orden visual
    #     return float(np.average(v[m], weights=w[m]))

    def _effective_dims(j: pd.DataFrame, f: dict) -> dict:
        # lee selección actual y, si hay SKU, infiere cat/sub/marca reales desde j
        dims = {
            "cat": f.get("cat", "All"),
            "sub": f.get("sub", "All"),
            "marca": f.get("marca", "All"),
            "sku": f.get("sku", "All"),
        }
        # Normalizados (por si no hay SKU)
        dims_key = {
            "cat": _norm_val(dims["cat"] if dims["cat"] != "All" else ""),
            "sub": _norm_val(dims["sub"] if dims["sub"] != "All" else ""),
            "marca": _norm_val(dims["marca"] if dims["marca"] != "All" else ""),
            "sku": _norm_val(dims["sku"] if dims["sku"] != "All" else ""),
        }

        # Si seleccionó SKU, inferimos claves desde la tabla mt (joined)
        if dims["sku"] != "All" and "SKU_KEY" in j.columns:
            row = j.loc[j["SKU_KEY"] == _norm_val(dims["sku"])].head(1)
            if len(row):
                if "CATEGORIA_KEY" in row.columns:  dims_key["cat"]   = row["CATEGORIA_KEY"].iloc[0]
                if "SUBCATEGORIA_KEY" in row.columns: dims_key["sub"]   = row["SUBCATEGORIA_KEY"].iloc[0]
                if "MARCA_KEY" in row.columns:      dims_key["marca"] = row["MARCA_KEY"].iloc[0]
                dims_key["sku"] = row["SKU_KEY"].iloc[0]
        return dims_key

    def _mask_level(j: pd.DataFrame, dims_key: dict, level: str) -> pd.Series:
        m = pd.Series(True, index=j.index)
        if level in ("cat", "sub", "marca", "sku") and "CATEGORIA_KEY" in j.columns and dims_key["cat"] != "ALL":
            m &= (j["CATEGORIA_KEY"] == dims_key["cat"])
        if level in ("sub", "marca", "sku") and "SUBCATEGORIA_KEY" in j.columns and dims_key["sub"] != "ALL":
            m &= (j["SUBCATEGORIA_KEY"] == dims_key["sub"])
        if level in ("marca", "sku") and "MARCA_KEY" in j.columns and dims_key["marca"] != "ALL":
            m &= (j["MARCA_KEY"] == dims_key["marca"])
        if level == "sku" and "SKU_KEY" in j.columns and dims_key["sku"] != "ALL":
            m &= (j["SKU_KEY"] == dims_key["sku"])
        return m

    def _wavg_metric_at_level(j: pd.DataFrame, metric: str, f: dict, level_code: str):
        if j.empty or metric not in j.columns:
            return np.nan
        dims_key = _effective_dims(j, f)
        sel = _mask_level(j, dims_key, level_code)
        vv = pd.to_numeric(j.loc[sel, metric], errors="coerce")
        ww = pd.to_numeric(j.loc[sel, "W_CU"], errors="coerce").clip(lower=0)
        mask = vv.notna() & ww.notna() & (ww > 0)
        if not mask.any():
            return np.nan
        return float(np.average(vv[mask], weights=ww[mask]))
    
    @reactive.Calc
    def transfer_ctx():
        """Une mapping con nuestro universo (último año) y calcula transferencias ponderadas por CU."""
        j = vtm_context()["joined"].copy()              # contiene SKU, CATEGORIA/SUBCATEGORIA/MARCA y W_CU
        if j.empty: 
            return {"flows": pd.DataFrame(), "universe": j}

        m = df_vtm_map().copy()
        if m.empty:
            return {"flows": pd.DataFrame(), "universe": j}

        # Unimos SOLO cuando la fuente es un SKU nuestro
        left = (m.merge(j[["SKU","SKU_KEY","CATEGORIA_KEY","SUBCATEGORIA_KEY","MARCA_KEY","W_CU"]],
                        left_on="SOURCE_SKU_KEY", right_on="SKU_KEY", how="inner",
                        suffixes=("","")))
        # Outbound a competencia (si quieres incluir canibalización interna, quita este filtro)
        left = left[ left["IS_SRC_COCA"] & (~left["IS_DEST_COCA"]) ].copy()

        # Transferencia esperada en CU (volumen * ratio)
        left["TRANSFER_CU"] = pd.to_numeric(left["W_CU"], errors="coerce") * pd.to_numeric(left["VOLUME_TRANSFER"], errors="coerce")
        left["TRANSFER_CU"] = left["TRANSFER_CU"].clip(lower=0)

        # Llaves de dims de la FUENTE (nuestro SKU)
        left.rename(columns={
            "CATEGORIA_KEY":"SRC_CAT_KEY",
            "SUBCATEGORIA_KEY":"SRC_SUB_KEY",
            "MARCA_KEY":"SRC_BRAND_KEY",
            "SKU":"SRC_SKU",
        }, inplace=True)

        return {"flows": left, "universe": j}
    
        # ========= Elasticidad vs VTM (cálculos auxiliares) =========

    def _is_coca(x: pd.Series) -> pd.Series:
        """Heurística robusta para detectar COCA COLA FABRICANTE en columnas de MFG."""
        s = _norm_series(x.fillna(""))
        return s.str.contains("COCA") & s.str.contains("FABRICANTE")

    @reactive.Calc
    def ev_monthly_flows():
        """
        Calcula flows mensuales (último año) distribuyendo VOLUME_TRANSFER por el CU mensual.
        Retorna DF con columnas:
        PERIOD_M, TYPE (Interna|Competencia), TRANSFER_CU, CU_TOTAL, EXT_SHARE
        Respetando los filtros (cat/sub/marca/sku) aplicados sobre el SKU fuente.
        """
        # 1) Universo último año y pesos / jerarquías
        ctx = vtm_context()
        d_last = ctx["last"].copy()          # CU mensual por SKU (último año)
        j      = ctx["joined"].copy()        # MT consolidated + W_CU + claves

        if d_last.empty or j.empty:
            return pd.DataFrame(columns=["PERIOD_M","TYPE","TRANSFER_CU","CU_TOTAL","EXT_SHARE"])

        # Periodo mensual
        d_last["PERIOD_M"] = (
            pd.to_datetime(d_last["PERIOD"], errors="coerce")
            .dt.to_period("M").dt.to_timestamp(how="start")
            .dt.tz_localize(None)
        )
        # 2) Filtros del usuario aplicados sobre SKU fuente (vía jerarquías en 'j')
        f = filters()
        dims_key = _effective_dims(j, f)
        mask_src = _mask_level(j, dims_key, "cat")  # empezamos con cat y refinamos abajo
        if dims_key["sub"] != "ALL":   mask_src &= (j["SUBCATEGORIA_KEY"] == dims_key["sub"])
        if dims_key["marca"] != "ALL": mask_src &= (j["MARCA_KEY"]        == dims_key["marca"])
        if dims_key["sku"] != "ALL":   mask_src &= (j["SKU_KEY"]          == dims_key["sku"])
        j_sel = j.loc[mask_src, ["SKU","SKU_KEY","CATEGORIA_KEY","SUBCATEGORIA_KEY","MARCA_KEY"]].drop_duplicates()

        if j_sel.empty:
            return pd.DataFrame(columns=["PERIOD_M","TYPE","TRANSFER_CU","CU_TOTAL","EXT_SHARE"])

        # 3) CU mensual del último año SOLO para los SKU fuente seleccionados
        d_last["SKU_KEY"] = _norm_series(d_last["SKU"])
        d_src = d_last.merge(j_sel[["SKU","SKU_KEY"]].drop_duplicates(), on="SKU_KEY", how="inner")

        if d_src.empty:
            return pd.DataFrame(columns=["PERIOD_M","TYPE","TRANSFER_CU","CU_TOTAL","EXT_SHARE"])

        # 4) Mapping UDM: unimos SOURCE_SKU -> DESTINATION_SKU/MFG y aplicamos ratio
        m = df_vtm_map().copy()
        if m.empty:
            # Sin mapping: devolvemos series vacías pero CU total para referencias
            cu_month = d_src.groupby("PERIOD_M", dropna=False)["CU"].sum(min_count=1).rename("CU_TOTAL").reset_index()
            out = cu_month.assign(TYPE="Competencia", TRANSFER_CU=0.0)
            out["EXT_SHARE"] = 0.0
            return out

        m["IS_SRC_COCA"]  = _is_coca(m.get("SRC_MFG",  pd.Series(index=m.index, dtype=str)))
        m["IS_DEST_COCA"] = _is_coca(m.get("DEST_MFG", pd.Series(index=m.index, dtype=str)))

        # Llaves de empalme robustas
        m["SOURCE_SKU_KEY"] = _norm_series(m["SOURCE_SKU"])
        d_src = d_src.merge(
            m[["SOURCE_SKU_KEY","DESTINATION_SKU","SRC_MFG","DEST_MFG","VOLUME_TRANSFER","IS_SRC_COCA","IS_DEST_COCA"]],
            left_on="SKU_KEY", right_on="SOURCE_SKU_KEY", how="left"
        )

        # 5) Tipo de flujo
        d_src["TYPE"] = np.where(
            (d_src["IS_SRC_COCA"] == True) & (d_src["IS_DEST_COCA"] == True), "Interna",
            np.where((d_src["IS_SRC_COCA"] == True) & (d_src["IS_DEST_COCA"] == False), "Competencia", "Otro")
        )
        d_src = d_src[d_src["TYPE"].isin(["Interna","Competencia"])].copy()

        # 6) Transferencia mensual estimada
        d_src["VOLUME_TRANSFER"] = pd.to_numeric(d_src["VOLUME_TRANSFER"], errors="coerce")
        d_src["CU"] = pd.to_numeric(d_src["CU"], errors="coerce")
        d_src["TRANSFER_CU"] = (d_src["CU"].clip(lower=0) * d_src["VOLUME_TRANSFER"].clip(lower=0)).fillna(0.0)

        # 7) Agregados mensuales
        flows = (d_src.groupby(["PERIOD_M","TYPE"], dropna=False)["TRANSFER_CU"]
                      .sum(min_count=1)
                      .reset_index())

        cu_month = (d_src.groupby(["PERIOD_M"], dropna=False)["CU"]
                         .sum(min_count=1)
                         .rename("CU_TOTAL")
                         .reset_index())

        out = flows.merge(cu_month, on="PERIOD_M", how="left")
        out["EXT_SHARE"] = np.where(out["CU_TOTAL"]>0,
                                    out["TRANSFER_CU"]/out["CU_TOTAL"],
                                    np.nan)
        out["PERIOD_M"] = (
            pd.to_datetime(out["PERIOD_M"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp(how="start")   # ← inicio de mes
            .dt.tz_localize(None)
        )
        out = out.sort_values(["PERIOD_M","TYPE"]).reset_index(drop=True)
        return out

    # ========= Elasticidad vs VTM (gráficos) =========

    @render_widget
    def fig_ev_correlation_timeline():
        df = ev_monthly_flows().copy()

        # Ventana del último año para garantizar eje X correcto
        win_start, win_end = last_full_year_window(df_base())
        # Índice mensual del año completo
        cal = pd.date_range(win_start, win_end - pd.offsets.MonthBegin(0), freq="MS")

        if df.empty:
            # Figura vacía pero con rango temporal correcto
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.update_xaxes(range=[cal.min(), cal.max()], type="date", tickformat="%b %Y")
            fig.update_layout(
                title="Transferencias mensuales (último año) vs Volumen y referencias de elasticidad",
                height=440, margin=dict(l=10,r=10,t=40,b=10),
                barmode="stack"
            )
            return fig

        # Pivot y reindex al calendario completo
        pivot = (df.pivot_table(index="PERIOD_M", columns="TYPE", values="TRANSFER_CU", aggfunc="sum")
                .reindex(cal).fillna(0.0))
        x = pivot.index

        cu_month = (df.groupby("PERIOD_M", dropna=False)["CU_TOTAL"].max()
                    .reindex(cal).fillna(0.0))

        # Referencias ponderadas
        ctx = vtm_context(); j = ctx["joined"]
        w_ppg = _wavg(j, "PPG_PE") if (j is not None and not j.empty) else np.nan
        w_net = _wavg(j, "NET_PE") if (j is not None and not j.empty) else np.nan

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "Interna" in pivot.columns:
            fig.add_trace(go.Bar(x=x, y=pivot["Interna"], name="Transferencia Interna"), secondary_y=False)
        if "Competencia" in pivot.columns:
            fig.add_trace(go.Bar(x=x, y=pivot["Competencia"], name="Transferencia a Competencia"), secondary_y=False)

        fig.add_trace(go.Scatter(x=x, y=cu_month, name="CU Total", mode="lines+markers"), secondary_y=True)

        if pd.notna(w_ppg):
            fig.add_hline(y=w_ppg, line_dash="dot", annotation_text="PPG_PE (ref)", annotation_position="top left")
        if pd.notna(w_net):
            fig.add_hline(y=w_net, line_dash="dash", annotation_text="NET_PE (ref)", annotation_position="bottom left")

        fig.update_layout(
            barmode="stack",
            title="Transferencias mensuales (último año) vs Volumen y referencias de elasticidad",
            height=440, margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(orientation="h", y=-0.2)
        )
        fig.update_yaxes(title_text="CU transferido (estimado)", secondary_y=False, showgrid=True, gridcolor=COCA_GRAY_LIGHT)
        fig.update_yaxes(title_text="CU Total", secondary_y=True, showgrid=False)
        fig.update_xaxes(range=[cal.min(), cal.max()], type="date", tickformat="%b %Y")
        return fig


    @render_widget
    def fig_ev_scatter_vtm_gap():
        """
        Scatter: VTM_RATIO vs brecha (PPG_PE - NET_PE), tamaño por W_CU.
        Cross-sectional en el último año; útil para ver correlación directa.
        """
        ctx = vtm_context()
        j = ctx["joined"].copy()
        if j.empty:
            return go.Figure()
        j["gap"] = pd.to_numeric(j["PPG_PE"], errors="coerce") - pd.to_numeric(j["NET_PE"], errors="coerce")
        j["W_CU"] = pd.to_numeric(j["W_CU"], errors="coerce").fillna(0)

        fig = px.scatter(
            j, x="VTM_RATIO", y="gap", size="W_CU",
            hover_data=["SKU","MARCA","SUBCATEGORIA","CATEGORIA"],
            title="VTM_RATIO vs (PPG_PE − NET_PE) — tamaño por volumen (último año)"
        )
        fig.add_hline(y=0, line_dash="dot")
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
        return fig

    @render_widget
    def fig_ev_stack_flows():
        df = ev_monthly_flows().copy()
        if df.empty:
            # fija el rango del último año aunque esté vacío
            win_start, win_end = last_full_year_window(df_base())
            cal = pd.date_range(win_start, win_end - pd.offsets.MonthBegin(0), freq="MS")
            fig = go.Figure()
            fig.update_layout(title="Transferencia estimada por mes — Interna vs Competencia",
                            height=420, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
            fig.update_xaxes(type="date", tickformat="%b %Y", range=[cal.min(), cal.max()])
            fig.update_yaxes(title_text="CU transferido")
            return fig

        # asegura fechas limpias
        df["PERIOD_M"] = pd.to_datetime(df["PERIOD_M"], errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["PERIOD_M"])

        df_bar = (df.groupby(["PERIOD_M","TYPE"], dropna=False)["TRANSFER_CU"]
                    .sum(min_count=1).reset_index().sort_values("PERIOD_M"))

        # rango del último año SIEMPRE
        win_start, win_end = last_full_year_window(df_base())
        cal = pd.date_range(win_start, win_end - pd.offsets.MonthBegin(0), freq="MS")

        fig = px.bar(
            df_bar, x="PERIOD_M", y="TRANSFER_CU", color="TYPE", barmode="stack",
            labels={"PERIOD_M":"Periodo","TRANSFER_CU":"CU transferido", "TYPE":"Tipo"},
            title="Transferencia estimada por mes — Interna vs Competencia"
        )
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
        fig.update_xaxes(type="date", tickformat="%b %Y", range=[cal.min(), cal.max()])
        return fig


    @render.download(filename="ev_monthly_flows.csv")
    def dl_ev_monthly_flows():
        df = ev_monthly_flows().copy()
        if df.empty:
            yield b""; return
        out = df.sort_values(["PERIOD_M","TYPE"]).copy()
        yield out.to_csv(index=False).encode("utf-8")


    # ===== Pestaña Insights: KPIs y gráficos
    # -------- KPIs fila 1: PPG_PE
    @render.ui
    def kpi_ppg_cat():
        if not _kpi_show_flags(filters())["cat"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "PPG_PE", filters(), "cat")
        return kpi_card("PPG_PE — Categoría", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_ppg_sub():
        if not _kpi_show_flags(filters())["sub"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "PPG_PE", filters(), "sub")
        return kpi_card("PPG_PE — Subcategoría", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_ppg_marca():
        if not _kpi_show_flags(filters())["marca"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "PPG_PE", filters(), "marca")
        return kpi_card("PPG_PE — Marca", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_ppg_sku():
        if not _kpi_show_flags(filters())["sku"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "PPG_PE", filters(), "sku")
        return kpi_card("PPG_PE — SKU", f"{val:.4f}" if pd.notna(val) else "—")

    # -------- KPIs fila 2: VTM_RATIO
    @render.ui
    def kpi_vtm_cat():
        if not _kpi_show_flags(filters())["cat"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "VTM_RATIO", filters(), "cat")
        return kpi_card("VTM Ratio — Categoría", f"{val:.3f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_vtm_sub():
        if not _kpi_show_flags(filters())["sub"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "VTM_RATIO", filters(), "sub")
        return kpi_card("VTM Ratio — Subcategoría", f"{val:.3f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_vtm_marca():
        if not _kpi_show_flags(filters())["marca"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "VTM_RATIO", filters(), "marca")
        return kpi_card("VTM Ratio — Marca", f"{val:.3f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_vtm_sku():
        if not _kpi_show_flags(filters())["sku"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "VTM_RATIO", filters(), "sku")
        return kpi_card("VTM Ratio — SKU", f"{val:.3f}" if pd.notna(val) else "—")

    # -------- KPIs fila 3: NET_PE
    @render.ui
    def kpi_net_cat():
        if not _kpi_show_flags(filters())["cat"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "NET_PE", filters(), "cat")
        return kpi_card("NET_PE — Categoría", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_net_sub():
        if not _kpi_show_flags(filters())["sub"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "NET_PE", filters(), "sub")
        return kpi_card("NET_PE — Subcategoría", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_net_marca():
        if not _kpi_show_flags(filters())["marca"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "NET_PE", filters(), "marca")
        return kpi_card("NET_PE — Marca", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_net_sku():
        if not _kpi_show_flags(filters())["sku"]:
            return ui.div(style="display:none;")
        j = vtm_context()["joined"]
        val = _wavg_metric_at_level(j, "NET_PE", filters(), "sku")
        return kpi_card("NET_PE — SKU", f"{val:.4f}" if pd.notna(val) else "—")


    @render_widget
    def fig_transfer_sankey():
        ctx = transfer_ctx()
        flows = ctx["flows"].copy()
        if flows.empty:
            return go.Figure()

        sel = _flows_mask_for_filters(flows, filters())
        flows = flows[sel].copy()
        if flows.empty:
            return go.Figure()

        # Agrega por fabricante destino (competencia) y toma Top N
        agg = (flows.groupby("DEST_MFG", dropna=False)["TRANSFER_CU"]
                    .sum(min_count=1)
                    .reset_index()
                    .sort_values("TRANSFER_CU", ascending=False))
        if agg.empty:
            return go.Figure()

        topN = 5
        top = agg.head(topN)
        others = agg.iloc[topN:]["TRANSFER_CU"].sum()
        if others > 0:
            top = pd.concat([top, pd.DataFrame({"DEST_MFG": ["OTROS"], "TRANSFER_CU": [others]})],
                            ignore_index=True)

        source_label = "Portafolio filtrado"
        nodes = [source_label] + top["DEST_MFG"].astype(str).tolist()
        idx = {lbl: i for i, lbl in enumerate(nodes)}

        links = dict(
            source=[idx[source_label]] * len(top),
            target=[idx[x] for x in top["DEST_MFG"].astype(str)],
            value=top["TRANSFER_CU"].tolist(),
            label=top["DEST_MFG"].astype(str).tolist(),
        )

        sank = go.Sankey(
            orientation="v",                     # ← vertical
            arrangement="snap",
            node=dict(
                label=nodes,
                pad=14,
                thickness=18,
                line=dict(color="#bbb", width=0.5),
            ),
            link=links,
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
        )

        fig = go.Figure(sank)

        # Altura dinámica para que quepan más etiquetas
        n_targets = len(top)
        h = max(420, min(220 + 44 * n_targets, 1000))

        fig.update_layout(
            title="Volume transfer → Competencia (ponderado por CU último año)",
            height=h,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        return fig

    
    @render_widget
    def fig_transfer_topdest():
        ctx = transfer_ctx()
        flows = ctx["flows"].copy()
        if flows.empty: return go.Figure()
        sel = _flows_mask_for_filters(flows, filters())
        flows = flows[sel].copy()
        if flows.empty: return go.Figure()

        # Top SKUs destino (competencia)
        dest = (flows.groupby(["DESTINATION_SKU","DEST_MFG"], dropna=False)["TRANSFER_CU"]
                    .sum(min_count=1)
                    .reset_index()
                    .sort_values("TRANSFER_CU", ascending=False)
                )
        dest = dest.head(20)
        if dest.empty: return go.Figure()

        fig = px.bar(
            dest.sort_values("TRANSFER_CU"),
            x="TRANSFER_CU", y="DESTINATION_SKU", orientation="h",
            color="DEST_MFG",
            title="Top SKUs de competencia que capturan volumen (último año)",
            labels={"TRANSFER_CU":"CU transferido estimado", "DESTINATION_SKU":"SKU destino", "DEST_MFG":"Fabricante"}
        )
        fig.update_layout(height=460, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
        return fig



    @render_widget
    def fig_eps_scatter():
        j = vtm_context()["joined"].copy()
        if j.empty: return go.Figure()
        j["W_CU"] = j["W_CU"].fillna(0)
        fig = px.scatter(
            j, x="PPG_PE", y="NET_PE", size="W_CU", color="VTM_RATIO",
            hover_data=["SKU","MARCA","SUBCATEGORIA","CATEGORIA"],
            title="Bruta (PPG_PE) vs Neta (NET_PE) — tamaño por volumen último año"
        )
        fig.add_trace(go.Scatter(x=[-5,5], y=[-5,5], mode="lines", name="y=x", line=dict(dash="dash")))
        fig.update_layout(height=CHART_H, margin=dict(l=10,r=10,t=40,b=10))
        return fig

    @render.download(filename="elasticidad_vtm_insights.csv")
    def dl_vtm():
        j = vtm_context()["joined"].copy()
        if j.empty: yield b""; return
        yield j.to_csv(index=False).encode("utf-8")

    # ===== TAB 1 KPIs (último año y sin 'modelo')
    @render.ui
    def kpi_1():
        base = df_joined().copy()
        f = filters()
        if f['cat'] != "All" and "CATEGORIA" in base.columns: base = base[base["CATEGORIA"] == f['cat']]
        if f['sub'] != "All" and "SUBCATEGORIA" in base.columns: base = base[base["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA" in base.columns: base = base[base["MARCA"] == f['marca']]
        if f['sku'] != "All" and "SKU" in base.columns: base = base[base["SKU"] == f['sku']]
        # Ventana último año
        win_start, win_end = last_full_year_window(df_base())
        base = base[(base["PERIOD"] >= win_start) & (base["PERIOD"] < win_end)].copy()
        if base.empty: return kpi_card("Meses con elasticidad calculada", "—")

        for c in ["dln_precio", "dln_cu", "FINAL_PE", "CU"]:
            if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")
        base = base.sort_values(["SKU","PERIOD"]).copy()
        base["CU_lag"] = base.groupby("SKU", sort=False)["CU"].shift(1)
        base["PERIOD_M"] = pd.to_datetime(base["PERIOD"]).dt.to_period("M").dt.to_timestamp()

        usar_ajustada = (input.modo_eps() == "Teórica Ajustada")
        if usar_ajustada:
            work = apply_ranges_block(base.copy()); col_eps = "epsilon_used"
        else:
            if "epsilon_teor" not in base.columns:
                dln_p = pd.to_numeric(base.get("dln_precio"), errors="coerce")
                dln_q = pd.to_numeric(base.get("dln_cu"), errors="coerce")
                base["epsilon_teor"] = (dln_q / dln_p).replace([np.inf, -np.inf], np.nan)
            work = base.copy(); col_eps = "epsilon_teor"

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "SKU": "SKU"}
        serie = monthly_eps_weighted(work, col_eps, entity_map.get(level_sel()))
        return kpi_card("Meses con elasticidad calculada", f"{int(serie['eps'].notna().sum()):,}")

    @render.ui
    def kpi_2():
        # PPG_PE ponderada (último año)
        ctx = vtm_context()
        j = ctx["joined"]
        if j.empty: return kpi_card("PPG_PE ponderada (bruta)", "—")
        val = _wavg(j, "PPG_PE")
        return kpi_card("PPG_PE ponderada (bruta)", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_3():
        # NET_PE ponderada (último año)
        ctx = vtm_context()
        j = ctx["joined"]
        if j.empty: return kpi_card("NET_PE ponderada (neta)", "—")
        val = _wavg(j, "NET_PE")
        return kpi_card("NET_PE ponderada (neta)", f"{val:.4f}" if pd.notna(val) else "—")

    # ===== TAB 1 chart (último año; líneas de referencia PPG_PE / NET_PE)
    @render_widget
    def fig_linea():
        base = df_joined().copy()
        f = filters()
        if f['cat'] != "All" and "CATEGORIA" in base.columns: base = base[base["CATEGORIA"] == f['cat']]
        if f['sub'] != "All" and "SUBCATEGORIA" in base.columns: base = base[base["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA" in base.columns: base = base[base["MARCA"] == f['marca']]
        if f['sku'] != "All" and "SKU" in base.columns: base = base[base["SKU"] == f['sku']]

        # Ventana último año completa
        win_start, win_end = last_full_year_window(df_base())
        base = base[(base["PERIOD"] >= win_start) & (base["PERIOD"] < win_end)].copy()
        if base.empty: return go.Figure()

        for c in ["dln_precio", "dln_cu", "FINAL_PE", "CU"]:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")
        base = base.sort_values(["SKU","PERIOD"]).copy()
        base["CU_lag"] = base.groupby("SKU", sort=False)["CU"].shift(1)
        base["PERIOD_M"] = pd.to_datetime(base["PERIOD"]).dt.to_period("M").dt.to_timestamp()

        usar_ajustada = (input.modo_eps() == "Teórica Ajustada")
        if usar_ajustada:
            work = apply_ranges_block(base.copy()); col_eps = "epsilon_used"; linea_label = "Elasticidad Teórica Ajustada"
        else:
            if "epsilon_teor" not in base.columns:
                dln_p = pd.to_numeric(base.get("dln_precio"), errors="coerce")
                dln_q = pd.to_numeric(base.get("dln_cu"), errors="coerce")
                base["epsilon_teor"] = (dln_q / dln_p).replace([np.inf, -np.inf], np.nan)
            work = base.copy(); col_eps = "epsilon_teor"; linea_label = "Elasticidad Teórica"

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "SKU": "SKU"}
        serie = monthly_eps_weighted(work, col_eps, entity_map.get(level_sel()))

        # Referencias PPG_PE & NET_PE ponderadas en el mismo último año y con filtros
        ctx = vtm_context()
        j = ctx["joined"]
        w_ppg = _wavg(j, "PPG_PE") if not j.empty else np.nan
        w_net = _wavg(j, "NET_PE") if not j.empty else np.nan

        # Eje X
        serie["PERIOD_M"] = pd.to_datetime(serie["PERIOD_M"], errors="coerce")
        x_dates = serie["PERIOD_M"].dt.tz_localize(None).dt.to_pydatetime()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_dates, y=serie["eps"],
            mode="lines+markers", name=linea_label, line=dict(width=2, color=COCA_RED),
            hovertemplate="%{x|%b %Y}<br>ε: %{y:.4f}<extra></extra>"
        ))
        if pd.notna(w_ppg):
            fig.add_hline(y=w_ppg, line_width=2, line_dash="dot",
                          annotation_text="PPG_PE (bruta)", annotation_position="top left")
        if pd.notna(w_net):
            fig.add_hline(y=w_net, line_width=2, line_dash="dash",
                          annotation_text="NET_PE (neta)", annotation_position="bottom left")

        fig.add_hline(y=0, line_width=1, line_dash="dot")
        fig.update_xaxes(type="date", tickformat="%b %Y", dtick="M1")
        fig.update_layout(
            title=f"{linea_label} (último año) — Nivel: {level_sel()}",
            yaxis_title="Elasticidad",
            hovermode="x unified", height=CHART_H,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        return fig

    @render.download(
        filename=lambda: f"linea_tiempo_elasticidad_{'ajustada' if input.modo_eps()=='Teórica Ajustada' else 'teorica'}_{level_sel().lower()}.csv"
    )
    def dl_linea():
        base = df_joined().copy()
        f = filters()
        if f['cat'] != "All" and "CATEGORIA" in base.columns: base = base[base["CATEGORIA"] == f['cat']]
        if f['sub'] != "All" and "SUBCATEGORIA" in base.columns: base = base[base["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA" in base.columns: base = base[base["MARCA"] == f['marca']]
        if f['sku'] != "All" and "SKU" in base.columns: base = base[base["SKU"] == f['sku']]

        win_start, win_end = last_full_year_window(df_base())
        base = base[(base["PERIOD"] >= win_start) & (base["PERIOD"] < win_end)].copy()
        if base.empty: yield b""; return

        for c in ["dln_precio", "dln_cu", "FINAL_PE", "CU"]:
            if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")
        base = base.sort_values(["SKU","PERIOD"]).copy()
        base["CU_lag"] = base.groupby("SKU", sort=False)["CU"].shift(1)
        base["PERIOD_M"] = pd.to_datetime(base["PERIOD"]).dt.to_period("M").dt.to_timestamp()

        usar_ajustada = (input.modo_eps() == "Teórica Ajustada")
        if usar_ajustada:
            work = apply_ranges_block(base.copy()); col_eps = "epsilon_used"
        else:
            if "epsilon_teor" not in base.columns:
                dln_p = pd.to_numeric(base.get("dln_precio"), errors="coerce")
                dln_q = pd.to_numeric(base.get("dln_cu"), errors="coerce")
                base["epsilon_teor"] = (dln_q / dln_p).replace([np.inf, -np.inf], np.nan)
            work = base.copy(); col_eps = "epsilon_teor"

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "SKU": "SKU"}
        serie = monthly_eps_weighted(work, col_eps, entity_map.get(level_sel()))
        # refs
        ctx = vtm_context(); j = ctx["joined"]
        w_ppg = _wavg(j, "PPG_PE") if not j.empty else np.nan
        w_net = _wavg(j, "NET_PE") if not j.empty else np.nan

        out = serie.rename(columns={"PERIOD_M": "PERIOD", "eps": "eps_mensual"}).copy()
        out["PPG_PE_ref"] = w_ppg
        out["NET_PE_ref"] = w_net
        yield out.to_csv(index=False).encode("utf-8")

    # ================= TAB 3: Clasificación (sin cambios de lógica base)
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
        if dj5.empty: return go.Figure()
        for c in ["CU","FINAL_PE"]:
            if c in dj5.columns: dj5[c] = pd.to_numeric(dj5[c], errors="coerce")
        dj5 = dj5.sort_values(["SKU","PERIOD"]).copy()
        dj5["CU_lag"] = dj5.groupby("SKU", sort=False)["CU"].shift(1)
        lvl = level_sel()
        if lvl == "Total":
            entity_col = "CATEGORIA" if "CATEGORIA" in dj5.columns else "SKU";  nivel_txt = "Categoría" if entity_col == "CATEGORIA" else "SKU"
        elif lvl == "Categoría":
            entity_col = "SUBCATEGORIA" if "SUBCATEGORIA" in dj5.columns else "SKU"; nivel_txt = "Subcategoría" if entity_col == "SUBCATEGORIA" else "SKU"
        else:
            entity_col = "SKU"; nivel_txt = "SKU"
        if entity_col not in dj5.columns: return go.Figure()
        agg_df = (dj5.groupby([entity_col], dropna=False).apply(lambda g: wavg_safe(g["FINAL_PE"], g["CU_lag"])).reset_index(name="Elasticidad_prom"))
        agg_df["Clase"] = agg_df["Elasticidad_prom"].apply(classify_elasticity)
        order_y = (agg_df[[entity_col, "Elasticidad_prom"]].sort_values("Elasticidad_prom", ascending=False)[entity_col].astype(str).tolist())
        color_map = {"Elástico (>1)": COCA_RED, "Unitario": COCA_BLACK, "Inelástico (<1)": COCA_GRAY_DARK}
        fig = go.Figure()
        for cls in ["Elástico (>1)", "Unitario", "Inelástico (<1)"]:
            sub = agg_df[agg_df["Clase"] == cls]
            if sub.empty: continue
            fig.add_trace(go.Bar(orientation="h", x=sub["Elasticidad_prom"], y=sub[entity_col].astype(str), name=cls, marker_color=color_map.get(cls, "#B3B3B3")))
        fig.update_layout(title=f"Elasticidad promedio por {nivel_txt}", xaxis_title="Elasticidad (ε)", yaxis_title=nivel_txt,
                          height=CHART_H, margin=dict(l=10, r=10, t=40, b=10), barmode="overlay",
                          yaxis=dict(categoryorder="array", categoryarray=order_y), legend=dict(orientation="h"))
        fig.add_vline(x=1, line_width=2, line_dash="dot", line_color="#999999")
        fig.add_vline(x=-1, line_width=2, line_dash="dot", line_color="#999999")
        return fig

    @render_widget
    def fig_pie():
        dj5 = df_joined().copy()
        if dj5.empty: return go.Figure()
        dj5["CU_lag"] = dj5.groupby("SKU", sort=False)["CU"].shift(1)
        agg_df = (dj5.groupby(["SKU"], dropna=False).apply(lambda g: wavg_safe(g["FINAL_PE"], g["CU_lag"])).reset_index(name="Elasticidad_prom"))
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
        if dj.empty: yield b""; return
        yield dj.to_csv(index=False).encode("utf-8")

    # ================= TAB 4: Explorador
    @render_widget
    def fig_explorer():
        d = df_filtered().copy()
        metric_mode = input.metric_mode()
        metric_col = "CU" if metric_mode == "Volumen (CU)" else ("INGRESO" if metric_mode == "Ventas ($ Ingreso Neto)" else "CU")
        if metric_mode == "Precio vs. CU (Δ%)":
            pcol, fallback_series = detect_price_col(d)
            if pcol is None: return go.Figure()
            if pcol == "__PRICE_FALLBACK__": d[pcol] = fallback_series
            if "INGRESO" in d.columns and "CU" in d.columns:
                dm = d.groupby(["PERIOD"], dropna=False).agg(CU=("CU","sum"), INGRESO=("INGRESO","sum")).reset_index()
                dm["PRICE_MEAN"] = np.where(dm["CU"]>0, dm["INGRESO"]/dm["CU"], np.nan)
            else:
                dm = d.groupby(["PERIOD"], dropna=False).agg(CU=("CU","sum"), PRICE_MEAN=(pcol,"mean")).reset_index()
            dm = dm.sort_values("PERIOD").copy()
            dm["dln_price"] = safe_lndiff(dm["PRICE_MEAN"]); dm["dln_cu"] = safe_lndiff(dm["CU"])
            dm["Δ Precio (%)"] = np.expm1(dm["dln_price"]) * 100; dm["Δ CU (%)"] = np.expm1(dm["dln_cu"]) * 100
            dm["PRICE_MEAN_prev"] = dm["PRICE_MEAN"].shift(1); dm["CU_prev"] = dm["CU"].shift(1)
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
                              xaxis=dict(showgrid=False), height=CHART_H, margin=dict(l=10,r=10,t=30,b=10))
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
            fig = px.line(agg, x="PERIOD", y=metric_col, color=(entity_col if entity_col in (agg.columns) else None),
                          title=f"{lvl} mensual — {titulo_metric}")
            fig.update_traces(mode="lines", line=dict(width=2), hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m}: %{y:,}")
            fig.update_layout(title=dict(x=0.01), hovermode="x unified", showlegend=False, xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT), height=CHART_H, margin=dict(l=10,r=10,t=30,b=10))
            return fig

    @render.download(filename=lambda: f"{level_sel()}_{'cu' if input.metric_mode()=='Volumen (CU)' else 'ingreso_neto'}.csv")
    def dl_explorer():
        d = df_filtered().copy()
        yield d.to_csv(index=False).encode("utf-8")

    # ================= TAB 6: Bennett
    @render_widget
    def fig_bennett():
        d_rev = df_filtered().copy()
        if not {"INGRESO","CU"}.issubset(set(d_rev.columns)):
            return go.Figure()
        d_rev = d_rev.sort_values(["SKU","PERIOD"]).copy()
        d_rev["P"] = np.where(pd.to_numeric(d_rev["CU"], errors="coerce")>0,
                              pd.to_numeric(d_rev["INGRESO"], errors="coerce")/pd.to_numeric(d_rev["CU"], errors="coerce"),
                              np.nan)
        d_rev["P_lag"] = d_rev.groupby("SKU")["P"].shift(1)
        d_rev["Q_lag"] = d_rev.groupby("SKU")["CU"].shift(1)
        d_rev["P_diff"] = d_rev["P"] - d_rev["P_lag"]
        d_rev["Q_diff"] = d_rev["CU"] - d_rev["Q_lag"]
        df_ben = (d_rev.assign(price_eff = d_rev["Q_lag"] * d_rev["P_diff"],
                               vol_eff   = d_rev["P_lag"] * d_rev["Q_diff"],
                               mix_eff   = d_rev["P_diff"] * d_rev["Q_diff"])
                  .groupby("PERIOD", dropna=False)[["price_eff","vol_eff","mix_eff"]]
                  .sum(min_count=1).reset_index().sort_values("PERIOD"))
        df_long = df_ben.melt(id_vars="PERIOD", var_name="Componente", value_name="ΔIngresos")
        names_map = {"price_eff":"Precio", "vol_eff":"Volumen", "mix_eff":"Mix"}
        df_long["Componente"] = df_long["Componente"].map(names_map)
        fig = px.bar(df_long, x="PERIOD", y="ΔIngresos", color="Componente", barmode="relative",
                     title="ΔIngresos por componente (Bennett)")
        fig.update_layout(height=440, margin=dict(l=10,r=10,t=40,b=10), legend=dict(orientation="h"))
        fig.add_hline(y=0, line_width=3, line_dash="solid", line_color="black")
        return fig

    @render.download(filename="descomposicion_ingresos_bennett.csv")
    def dl_bennett():
        d = df_filtered().copy()
        yield d.to_csv(index=False).encode("utf-8")

app = App(app_ui, server)
