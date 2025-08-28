import os
import io
import sys
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit.components.v1 import html as st_html
from textwrap import dedent

# =====================
# CONFIGURACIÓN Y ESTILO GLOBAL
# =====================
st.set_page_config(page_title="Elasticity Price Monitor", layout="wide")

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap" rel="stylesheet">
<style>
    [data-testid="stMetric"] { padding: 4px 8px !important; margin: 0 0 6px 0 !important; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; }
    [data-testid="stMetricValue"] { font-size: 20px !important; line-height: 1.2 !important; }
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    div[data-testid="stDecoration"], div[data-testid="stStatusWidget"], div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; background: transparent !important; }
    header { display: none !important; }
    section.main > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
    [data-testid="stPlotlyChart"] { margin-top: 20px !important; }
    .block-container { padding-top: 0.2rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; border-bottom: 2px solid #E9ECEF; margin: 0 0 4px 0 !important; }
    .stTabs [role="tab"] { font-weight: 600; color: #4B4B4B; }
    .stTabs [aria-selected="true"] { color: #E41E26 !important; border-bottom: 3px solid #E41E26 !important; }
    .stDownloadButton button, .stButton button {
        background-color: #E41E26 !important; color: #FFFFFF !important;
        border-radius: 10px !important; border: 2px solid #E41E26 !important; font-weight: 700 !important;
    }
    .card-cc { background: #E41E26; color: #FFFFFF; border-radius: 12px; border: 2px solid #E41E26; box-shadow: 0 2px 8px rgba(0,0,0,.06); padding: 10px 12px; }
    .card-cc h4 { margin: 0 0 6px 0; font-size: 14px; font-weight: 700; }
    .card-cc ul { list-style: none; padding-left: 0; margin: 6px 0 0 0; }
    .card-cc li { margin: 0 0 8px 0; line-height: 1.25; }
    .pill-share { background: #FFFFFF; color: #E41E26; border-radius: 10px; padding: 2px 6px; font-size: 11px; font-weight: 700; }
    .muted { color: rgba(255,255,255,.9); font-size: 12px; }
</style>
"""
st_html(CSS, height=0)

# ===== THEME =====
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

def kpi_card(label, value, variant="primary", size="compact"):
    if variant == "primary":
        bg, fg, border = COCA_RED, COCA_WHITE, COCA_RED
    else:
        bg, fg, border = COCA_WHITE, COCA_BLACK, COCA_RED

    if size == "compact":
        pad = "10px 12px"; fs_label = "11px"; fs_value = "22px"; radius = "12px"
    else:
        pad = "16px 18px"; fs_label = "12px"; fs_value = "32px"; radius = "16px"

    html = f"""
    <div style="background:{bg};color:{fg};padding:{pad};border-radius:{radius};
                border:2px solid {border};box-shadow:0 2px 8px rgba(0,0,0,.06);height:100%;">
        <div style="font-size:{fs_label}; letter-spacing:.4px; opacity:.9; margin-bottom:4px;">{label}</div>
        <div style="font-size:{fs_value}; font-weight:700; line-height:1">{value}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def fmt_int(n):
    try:
        return f"{int(round(n)):,}"
    except Exception:
        return "-"

# =====================
# DATA SOURCES (LOCAL)
# =====================
DATA_DIR = os.path.join(os.getcwd(), "data")
FILE_SERIES = ("prediction_input_series.csv", "prediction_input_series.parquet")
FILE_ELAST  = ("final_pe_mt.csv", "final_pe_mt.parquet")
FILE_CONS   = ("mt_consolidated_pe.csv", "mt_consolidated_pe.parquet")

@st.cache_data(show_spinner=True)
def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    raise ValueError(f"Formato no soportado: {path}")

def _try_load_from_folder(fname_candidates) -> pd.DataFrame | None:
    for fname in fname_candidates:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            return _read_any(fpath)
    return None

# =====================
# DATOS (sidebar con expander)
# =====================
st.sidebar.markdown(
    f"<h3 style='color:{COCA_BLACK}; margin-bottom:4px;'>Datos</h3>",
    unsafe_allow_html=True
)

with st.sidebar:
    with st.expander("📂 Datos (haz clic para desplegar)", expanded=False):
        st.markdown("Carga desde <code>/data</code> o súbelos aquí:", unsafe_allow_html=True)

        up_series = st.file_uploader(
            "prediction_input_series (.csv/.parquet)",
            type=["csv", "parquet"],
            key="prediction_input_series"
        )
        up_elast = st.file_uploader(
            "final_pe_mt (.csv/.parquet)",
            type=["csv", "parquet"],
            key="final_pe_mt"
        )
        up_cons = st.file_uploader(
            "mt_consolidated_pe (.csv/.parquet)",
            type=["csv", "parquet"],
            key="mt_consolidated_pe"
        )

def _read_uploaded(up):
    if up is None: return None
    name = up.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(up)
    elif name.endswith(".parquet"):
        data = up.read()
        return pd.read_parquet(io.BytesIO(data))
    else:
        st.error(f"Formato no soportado: {name}")
        return None

# =====================
# CARGADORES
# =====================
def pick_col(df, opciones):
    for c in opciones:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan]*len(df), index=df.index)

@st.cache_data(show_spinner=True)
def load_series(df_series_input: pd.DataFrame | None = None) -> pd.DataFrame:
    df_series = df_series_input
    if df_series is None:
        df_series = _try_load_from_folder(FILE_SERIES)
    if df_series is None:
        raise FileNotFoundError("No encontré `prediction_input_series` en /data ni fue subido.")

    df_series = df_series.copy()

    # --- Normalización PERIOD ---
    if "PERIOD" in df_series.columns:
        s = df_series["PERIOD"].astype(str).str.strip()
        periods = pd.to_datetime(s, format="%Y_%m", errors="coerce")
        if periods.isna().any():
            m1 = s.str.match(r"^\d{4}-\d{1,2}$")
            periods.loc[m1] = pd.to_datetime(s[m1] + "-01", errors="coerce")
            m2 = s.str.match(r"^\d{6}$")
            periods.loc[m2] = pd.to_datetime(s[m2], format="%Y%m", errors="coerce")
            m3 = s.str.match(r"^\d{8}$")
            periods.loc[m3] = pd.to_datetime(s[m3], format="%Y%m%d", errors="coerce")
        df_series["PERIOD"] = periods
    else:
        y = pick_col(df_series,["YEAR", "year"])
        m = pick_col(df_series,["MONTH", "month"])
        y_str = y.astype("Int64").astype(str).str.zfill(4)
        m_str = m.astype("Int64").astype(str).str.zfill(2)
        df_series["PERIOD"] = pd.to_datetime(y_str + "-" + m_str + "-01", errors="coerce")

    # --- Normalizaciones de nombres ---
    rename_map = {}
    if "CATEGORY_REVISED"      in df_series.columns: rename_map["CATEGORY_REVISED"] = "CATEGORIA"
    if "SUBCATEGORY_REVISED"   in df_series.columns: rename_map["SUBCATEGORY_REVISED"] = "SUBCATEGORIA"
    if "SKU_GROUP"             in df_series.columns: rename_map["SKU_GROUP"] = "SKU"
    if "UC_CALCULATED"         in df_series.columns: rename_map["UC_CALCULATED"] = "UC"
    if "INGRESO_NETO" in df_series.columns or "INGRES_NETO" in df_series.columns:
        rename_map["INGRESO_NETO" if "INGRESO_NETO" in df_series.columns else "INGRES_NETO"] = "INGRESO"
    if "PRICE"                 in df_series.columns: rename_map["PRICE"] = "PRECIO"
    if rename_map:
        df_series = df_series.rename(columns=rename_map)

    for need in ["PERIOD", "SKU"]:
        if need not in df_series.columns:
            raise KeyError(f"Falta la columna requerida: {need}")

    for col in ["UC","INGRESO"]:
        if col in df_series.columns:
            df_series[col] = pd.to_numeric(df_series[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

    return df_series

@st.cache_data(show_spinner=True)
def load_joined_for_elasticity(
    df_base: pd.DataFrame,
    df_elast_input: pd.DataFrame | None = None,
    df_cons_input: pd.DataFrame | None = None
) -> pd.DataFrame:
    pe = df_elast_input
    if pe is None:
        pe = _try_load_from_folder(FILE_ELAST)
    if pe is None:
        raise FileNotFoundError("No encontré `final_pe_mt` en /data ni fue subido.")

    pe = pe.rename(columns={
        "SKU_NAME": "SKU",
        "FINAL_PE": "FINAL_PE",
        "PE_RANGE_CHECK_OVERALL": "PE_RANGE_CHECK_OVERALL",
        "P1_MIN": "P1_MIN", "P1_MAX": "P1_MAX",
        "P2_MIN": "P2_MIN", "P2_MAX": "P2_MAX",
        "FINAL_PE_SOURCE": "FINAL_PE_SOURCE",
    })

    dj = df_base.merge(pe, on="SKU", how="left").sort_values(["SKU", "PERIOD"]).copy()

    mt_pe = df_cons_input
    if mt_pe is None:
        mt_pe = _try_load_from_folder(FILE_CONS)
    if mt_pe is None:
        mt_pe = pd.DataFrame()

    if not mt_pe.empty:
        right_key = "PPG" if "PPG" in mt_pe.columns else "SKU"
        dj = dj.merge(mt_pe, how="left", left_on="SKU", right_on=right_key)

    if "NET_PE" in dj.columns:
        dj["FINAL_PE"] = pd.to_numeric(dj["NET_PE"], errors="coerce")

    price_col = next((c for c in ["PRECIO", "PRICE"] if c in dj.columns), None)
    uc_col    = "UC" if "UC" in dj.columns else None
    missing = []
    if price_col is None: missing.append("PRECIO/PRICE")
    if uc_col is None:    missing.append("UC")
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {', '.join(missing)}")

    dj[price_col] = pd.to_numeric(dj[price_col], errors="coerce")
    dj[uc_col]    = pd.to_numeric(dj[uc_col],    errors="coerce")

    dj["lag_price"] = dj.groupby("SKU")[price_col].shift(1)
    dj["lag_uc"]    = dj.groupby("SKU")[uc_col].shift(1)

    dj["dln_precio"] = np.nan
    ok_p = (dj[price_col] > 0) & (dj["lag_price"] > 0)
    dj.loc[ok_p, "dln_precio"] = np.log(dj.loc[ok_p, price_col]) - np.log(dj.loc[ok_p, "lag_price"])

    dj["dln_uc"] = np.nan
    ok_u = (dj[uc_col] > 0) & (dj["lag_uc"] > 0)
    dj.loc[ok_u, "dln_uc"] = np.log(dj.loc[ok_u, uc_col]) - np.log(dj.loc[ok_u, "lag_uc"])

    dj["impacto_esperado"] = dj["FINAL_PE"] * dj["dln_precio"]

    return dj

# =====================
# MÉTRICAS / CLIPPING / HELPERS
# =====================
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

def apply_ranges_block(dd):
    d = dd.copy()
    for c in ["P1_MIN", "P1_MAX", "P2_MIN", "P2_MAX", "PE_RANGE_CHECK_OVERALL"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    cond1 = (d.get("PE_RANGE_CHECK_OVERALL") == 1)
    cond2 = (d.get("PE_RANGE_CHECK_OVERALL") == 2)

    e_min = np.where(cond1, d.get("P1_MIN"), np.where(cond2, d.get("P2_MIN"), np.nan))
    e_max = np.where(cond1, d.get("P1_MAX"), np.where(cond2, d.get("P2_MAX"), np.nan))

    e_teor = pd.to_numeric(d.get("dln_uc"), errors="coerce") / pd.to_numeric(d.get("dln_precio"), errors="coerce")
    e_teor = e_teor.replace([np.inf, -np.inf], np.nan)

    eps_local = 1e-8
    dln_precio = pd.to_numeric(d.get("dln_precio"), errors="coerce")
    no_move = dln_precio.abs() < eps_local

    lower = np.where(np.isnan(e_min), -np.inf, e_min)
    upper = np.where(np.isnan(e_max),  np.inf,  e_max)
    e_used = np.clip(e_teor.to_numpy(), lower, upper)

    d["real_clipped"] = e_used * dln_precio.to_numpy()
    d.loc[no_move, "real_clipped"] = 0.0

    d["clip_source"] = None
    to_min = (e_teor.to_numpy() < lower) & ~no_move.to_numpy()
    to_max = (e_teor.to_numpy() > upper) & ~no_move.to_numpy()
    cond1_np = np.asarray(cond1.fillna(False)) if hasattr(cond1, "fillna") else np.zeros(len(d), dtype=bool)
    cond2_np = np.asarray(cond2.fillna(False)) if hasattr(cond2, "fillna") else np.zeros(len(d), dtype=bool)

    d.loc[to_min & cond1_np, "clip_source"] = "P1_MIN"
    d.loc[to_max & cond1_np, "clip_source"] = "P1_MAX"
    d.loc[to_min & cond2_np, "clip_source"] = "P2_MIN"
    d.loc[to_max & cond2_np, "clip_source"] = "P2_MAX"
    d.loc[no_move,           "clip_source"] = "NO_PRICE_MOVE"

    d["elasticidad_usada_real"] = e_used
    return d

def detect_price_col(df_):
    if "PRECIO" in df_.columns: return "PRECIO", None
    if "PRICE"  in df_.columns: return "PRICE",  None
    if ("INGRESO" in df_.columns) and ("UC" in df_.columns):
        s = np.where(
            pd.to_numeric(df_["UC"], errors="coerce") > 0,
            pd.to_numeric(df_["INGRESO"], errors="coerce") / pd.to_numeric(df_["UC"], errors="coerce"),
            np.nan
        )
        return "__PRICE_FALLBACK__", pd.Series(s, index=df_.index)
    return None, None

# ===== Helper seguro para promedio ponderado (evita ZeroDivisionError) =====
def wavg_safe(values, weights, fallback="mean"):
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
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

# ===== Log-diff seguro (definir ANTES de usarlo en Tab 1) =====
def safe_lndiff(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0, np.nan)
    return np.log(s).diff()

# =====================
# INGESTA: PRIORIDAD uploaders → /data
# =====================
df_series_up = _read_uploaded(up_series)
df_elast_up  = _read_uploaded(up_elast)
df_cons_up   = _read_uploaded(up_cons)

try:
    df = load_series(df_series_up)
except Exception as e:
    st.error(f"Error cargando prediction_input_series: {e}")
    st.stop()

# =====================
# SIDEBAR (FILTROS)
# =====================
st.sidebar.markdown(f"<h3 style='color:{COCA_BLACK}; margin-bottom:4px;'>Filtros</h3>", unsafe_allow_html=True)
metric_mode = st.sidebar.radio(
    "Métrica / Vista",
    ("Volumen (UC)", "Ventas ($ Ingreso)", "Precio vs. UC (Δ%)"),
    help="Selecciona qué ver en el panel principal."
)
metric_col = "UC" if metric_mode == "Volumen (UC)" else ("INGRESO" if metric_mode == "Ventas ($ Ingreso)" else "UC")

if metric_col not in df.columns:
    st.sidebar.warning(f"No encontré la columna '{metric_col}'. Intento usar otra disponible.")
    metric_col = "UC" if "UC" in df.columns else ("INGRESO" if "INGRESO" in df.columns else None)
if metric_col is None:
    st.error("No hay columnas de métricas disponibles (UC/INGRESO).")
    st.stop()

cats = ["All"] + sorted(df.get("CATEGORIA", pd.Series(dtype=str)).dropna().unique().tolist())
cat_sel = st.sidebar.selectbox("Categoría", cats)

if cat_sel != "All" and "SUBCATEGORIA" in df.columns:
    subs = ["All"] + sorted(df.loc[df["CATEGORIA"]==cat_sel,"SUBCATEGORIA"].dropna().unique().tolist())
else:
    subs = ["All"] + sorted(df.get("SUBCATEGORIA", pd.Series(dtype=str)).dropna().unique().tolist())
sub_sel = st.sidebar.selectbox("Subcategoría", subs)

if sub_sel != "All" and "SUBCATEGORIA" in df.columns:
    skus_base = df.loc[df["SUBCATEGORIA"]==sub_sel,"SKU"]
else:
    skus_base = df["SKU"]
skus = ['All'] + sorted(skus_base.dropna().unique().tolist())
sku_sel = st.sidebar.selectbox("SKU", skus)

# Fechas
periods = pd.to_datetime(df["PERIOD"], errors="coerce")
min_d, max_d = periods.min(), periods.max()
if pd.isna(min_d) or pd.isna(max_d):
    st.sidebar.warning("No hay fechas válidas en 'PERIOD'. Uso un rango por defecto.")
    min_d = pd.Timestamp("2021-01-01")
    max_d = pd.Timestamp.today().normalize()
default_range = [min_d.date(), max_d.date()]
dr = st.sidebar.date_input("Rango de Fechas", value=default_range, key="date_range")
if isinstance(dr, (list, tuple)) and len(dr) == 2:
    d_start, d_end = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
else:
    d_start = pd.Timestamp(dr); d_end = pd.Timestamp(dr)

def infer_level(cat_sel, sub_sel, sku_sel):
    if sku_sel != "All":
        return "SKU"
    elif sub_sel != "All":
        return "Subcategoría"
    elif cat_sel != "All":
        return "Categoría"
    else:
        return "Total"

level = infer_level(cat_sel, sub_sel, sku_sel)

topn = None
if level in ["Categoría", "Subcategoría", "SKU"]:
    topn = st.sidebar.slider(
        "Mostrar Top N (por volumen total)",
        5, 30, 15,
        help=f"Ordenado por {('UC' if metric_mode!='Ventas ($ Ingreso)' else 'Ingreso')} total en el rango seleccionado"
    )

chart_h = st.sidebar.slider(
    "Altura de gráficas (px)", 
    min_value=280, max_value=600, value=380, step=10,
    help="Ajusta la altura de todas las gráficas"
)

# =====================
# TABS
# =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sales Explorer", "Elasticity Performance","Elasticity Classification", "Diagnostics", "Revenue Decomposition"])

# =============== TAB 1: Sales Explorer
with tab1:
    d = df.copy()
    if cat_sel != "All" and "CATEGORIA" in d.columns:  d = d[d["CATEGORIA"] == cat_sel]
    if sub_sel != "All" and "SUBCATEGORIA" in d.columns: d = d[d["SUBCATEGORIA"] == sub_sel]
    if sku_sel != "All" and "SKU" in d.columns:          d = d[d["SKU"] == sku_sel]
    d = d[(d["PERIOD"] >= d_start) & (d["PERIOD"] <= d_end)]

    col1, col2, col3, col4 = st.columns([1,1,1,1])
    base_col_for_kpi = "UC" if metric_mode == "Precio vs. UC (Δ%)" else metric_col
    total_metric = pd.to_numeric(d[base_col_for_kpi], errors="coerce").sum(skipna=True)
    skus_activos = d["SKU"].nunique()
    meses = d["PERIOD"].dt.to_period("M").nunique() if len(d) else 0
    prom_mensual = (total_metric/meses) if meses else np.nan
    try:
        this_year = d["PERIOD"].max().year; prev_year = this_year - 1
        cur = pd.to_numeric(d[d["PERIOD"].dt.year == this_year][base_col_for_kpi], errors="coerce").sum()
        prev = pd.to_numeric(d[d["PERIOD"].dt.year == prev_year][base_col_for_kpi], errors="coerce").sum()
        yoy = ((cur - prev) / prev * 100) if prev and not np.isnan(prev) else np.nan
    except Exception:
        yoy = np.nan

    metric_label_for_kpi = ("Volumen (UC)" if metric_mode=="Precio vs. UC (Δ%)" else metric_mode)
    with col1: kpi_card(f"Total {metric_label_for_kpi}", fmt_int(total_metric), "primary", size="compact")
    with col2: kpi_card("Volumen Promedio Mensual", fmt_int(prom_mensual), "secondary", size="compact")
    with col3: kpi_card("SKU's Activos", fmt_int(skus_activos), "secondary", size="compact")
    with col4: kpi_card("Crecimiento vs Año Ant.", f"{yoy:.1f}%" if not np.isnan(yoy) else "-", "secondary", size="compact")

    st.markdown('<div class="vspace-12"></div>', unsafe_allow_html=True)

    # ====== VISTA ESPECIAL: Precio vs. UC (Δ%)
    if metric_mode == "Precio vs. UC (Δ%)":
        pcol, fallback_series = detect_price_col(d)
        if pcol is None:
            st.warning("No encontré columna de precio ni puedo derivarla (INGRESO/UC).")
            st.stop()
        if pcol == "__PRICE_FALLBACK__":
            d[pcol] = fallback_series

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "SKU": "SKU"}
        entity_col = entity_map[level]

        # --- Total (sin entidad) ---
        if entity_col is None:
            if "INGRESO" in d.columns and "UC" in d.columns:
                dm = d.groupby(["PERIOD"], dropna=False).agg(
                    UC=("UC", "sum"), INGRESO=("INGRESO","sum")
                ).reset_index()
                dm["PRICE_MEAN"] = np.where(dm["UC"]>0, dm["INGRESO"]/dm["UC"], np.nan)
            else:
                dm = d.groupby(["PERIOD"], dropna=False).agg(
                    UC=("UC","sum") if "UC" in d.columns else ("SKU","count"),
                    PRICE_MEAN=(pcol,"mean")
                ).reset_index()

            dm = dm.sort_values("PERIOD").copy()
            # ✅ log-diff seguro
            dm["dln_price"] = safe_lndiff(dm["PRICE_MEAN"])
            dm["dln_uc"]    = safe_lndiff(dm["UC"])
            dm["Δ Precio (%)"] = np.expm1(dm["dln_price"]) * 100
            dm["Δ UC (%)"]     = np.expm1(dm["dln_uc"]) * 100

            plot_df = dm[["PERIOD","Δ Precio (%)","Δ UC (%)"]].melt(
                id_vars="PERIOD", var_name="Serie", value_name="Valor"
            )

            fig = px.line(plot_df, x="PERIOD", y="Valor", color="Serie",
                          title="Δ% Mensual: Precio vs. UC — Total")
            fig.update_traces(mode="lines", line=dict(width=2),
                              hovertemplate="%{x|%Y-%m}<br>%{fullData.name}: %{y:.1f}%")
            fig.update_layout(
                colorway=[COCA_RED, "#1F1F1F"],
                title=dict(font=dict(color=COCA_BLACK, size=18), x=0.01),
                hovermode="x unified", showlegend=True,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT, title="Δ% vs mes anterior"),
                height=chart_h, margin=dict(l=10, r=10, t=30, b=10)
            )
            fig.add_hline(y=0, line_width=1, line_dash="dot")
            st.plotly_chart(fig, use_container_width=True)

            out = dm[["PERIOD","PRICE_MEAN","UC","Δ Precio (%)","Δ UC (%)"]].copy()
            st.download_button("Descargar Δ Precio vs. Δ UC (Total)",
                               out.to_csv(index=False).encode("utf-8"),
                               file_name="delta_precio_vs_uc_total.csv", type="primary")

        # --- Con entidad (Categoría/Subcategoría/SKU) ---
        else:
            if "INGRESO" in d.columns and "UC" in d.columns:
                g = d.groupby(["PERIOD", entity_col], dropna=False).agg(
                    UC=("UC","sum"), INGRESO=("INGRESO","sum")
                ).reset_index()
                g["PRICE_MEAN"] = np.where(g["UC"]>0, g["INGRESO"]/g["UC"], np.nan)
            else:
                g = d.groupby(["PERIOD", entity_col], dropna=False).agg(
                    UC=("UC","sum") if "UC" in d.columns else ("SKU","count"),
                    PRICE_MEAN=(pcol,"mean")
                ).reset_index()

            if topn is not None:
                top_entities = (g.groupby(entity_col)["UC"].sum()
                                  .sort_values(ascending=False).head(topn).index)
                g = g[g[entity_col].isin(top_entities)]

            g = g.sort_values([entity_col, "PERIOD"]).copy()
            # ✅ Usar transform para conservar índice (evita TypeError)
            g["dln_price"] = g.groupby(entity_col, group_keys=False)["PRICE_MEAN"].transform(safe_lndiff)
            g["dln_uc"]    = g.groupby(entity_col, group_keys=False)["UC"].transform(safe_lndiff)
            g["Δ Precio (%)"] = np.expm1(g["dln_price"]) * 100
            g["Δ UC (%)"]     = np.expm1(g["dln_uc"]) * 100

            plot_df = g.melt(
                id_vars=["PERIOD", entity_col],
                value_vars=["Δ Precio (%)","Δ UC (%)"],
                var_name="Serie", value_name="Valor"
            )

            fig = px.line(
                plot_df, x="PERIOD", y="Valor",
                color=entity_col, line_dash="Serie",
                title=f"Δ% Mensual: Precio vs. UC — {level}"
            )
            dash_map = {"Δ UC (%)": "solid", "Δ Precio (%)": "dash"}
            fig.for_each_trace(lambda tr: tr.update(
                line=dict(width=2, dash=dash_map.get(tr.name.split(", ")[-1], "solid"))
            ))

            fig.update_traces(hovertemplate="%{x|%Y-%m}<br>%{legendgroup}: %{y:.1f}% (%{line.dash})")
            fig.update_layout(
                title=dict(font=dict(color=COCA_BLACK, size=18), x=0.01),
                hovermode="x unified", showlegend=True,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT, title="Δ% vs mes anterior"),
                height=chart_h, margin=dict(l=10, r=10, t=30, b=10)
            )
            fig.add_hline(y=0, line_width=1, line_dash="dot")
            st.plotly_chart(fig, use_container_width=True)

            out_cols = ["PERIOD", entity_col, "PRICE_MEAN", "UC", "Δ Precio (%)", "Δ UC (%)"]
            st.download_button(
                f"Descargar Δ Precio vs. Δ UC ({level})",
                g[out_cols].to_csv(index=False).encode("utf-8"),
                file_name=f"delta_precio_vs_uc_{level.lower()}.csv",
                type="primary"
            )

    # ====== Vista estándar (UC/Ingreso)
    else:
        group = ["PERIOD"]; entity_col = None
        if level=="Categoría" and "CATEGORIA" in d.columns:
            group.append("CATEGORIA"); entity_col = "CATEGORIA"
        elif level=="Subcategoría" and "SUBCATEGORIA" in d.columns:
            group.append("SUBCATEGORIA"); entity_col = "SUBCATEGORIA"
        elif level=="SKU" and "SKU" in d.columns:
            group.append("SKU"); entity_col = "SKU"

        agg = d.groupby(group, dropna=False)[metric_col].sum().reset_index()
        left, right = st.columns([4, 1.35])

        with left:
            if entity_col and (topn is not None) and (entity_col in agg.columns):
                tops_for_plot = (agg.groupby(entity_col)[metric_col].sum()
                                    .sort_values(ascending=False)
                                    .head(topn).index)
                agg_plot = agg[agg[entity_col].isin(tops_for_plot)].copy()
            else:
                agg_plot = agg

            titulo_metric = "UC" if metric_col == "UC" else "Ingreso"
            title = f"{level} Mensual {titulo_metric}"
            fig = px.line(
                agg_plot, x="PERIOD", y=metric_col,
                color=(entity_col if (entity_col and entity_col in agg_plot.columns) else None),
                title=title
            )
            fig.update_traces(mode="lines", line=dict(width=2),
                              hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m}: %{y:,}")
            fig.update_layout(
                title=dict(font=dict(color=COCA_BLACK, size=18), x=0.01),
                hovermode="x unified", showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT),
                height=chart_h, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            csv = agg.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV", csv, file_name=f"{level}_{metric_col}.csv", type="primary")

        with right:
            display_col, etiqueta = None, None
            if level == "Total" and "CATEGORIA" in d.columns:
                display_col, etiqueta = "CATEGORIA", "Categoría"
            elif level == "Categoría" and "SUBCATEGORIA" in d.columns:
                display_col, etiqueta = "SUBCATEGORIA", "Subcategoría"
            elif level == "Subcategoría" and "SKU" in d.columns:
                display_col, etiqueta = "SKU", "SKU"
            else:
                display_col, etiqueta = None, None

            if display_col is not None and topn is not None and display_col in d.columns:
                st.markdown(
                    f"<div style='font-weight:700; color:{COCA_BLACK}; margin:4px 0 8px 0;'>Top {topn} {etiqueta}</div>",
                    unsafe_allow_html=True
                )

                total_sel = pd.to_numeric(d[metric_col], errors="coerce").sum()

                top_df = (
                    d.groupby(display_col, dropna=False)[metric_col]
                    .sum(min_count=1)
                    .sort_values(ascending=False)
                    .head(topn)
                    .reset_index()
                )
                top_df["__share"] = np.where(
                    total_sel > 0,
                    (pd.to_numeric(top_df[metric_col], errors="coerce") / total_sel) * 100.0,
                    np.nan
                )

                items = []
                for i, row in top_df.iterrows():
                    name = str(row[display_col])
                    val = row[metric_col]
                    share = row["__share"]
                    val_txt = f"{int(round(val)):,}" if pd.notna(val) else "-"
                    share_txt = (f"{share:.1f}%" if pd.notna(share) else "-")
                    items.append(
                        f"<li style='margin-bottom:8px;'>"
                        f"<span style='color:{COCA_RED}; font-weight:700;'>{i+1}.</span> {name}"
                        f"<br><span style='color:{COCA_GRAY_DARK}; font-size:12px;'>Total: {val_txt} "
                        f"| <span style='background:{COCA_RED}; color:#fff; border-radius:10px; padding:2px 6px; font-size:11px; font-weight:700;'>{share_txt}</span>"
                        f"</span></li>"
                    )
                ul = "<ul style='list-style:none; padding-left:0; margin-top:4px;'>" + "\n".join(items) + "</ul>"
                st.markdown(ul, unsafe_allow_html=True)
            else:
                st.write("")

# =============== TAB 2: Elasticity Performance
with tab2:
    try:
        dj = load_joined_for_elasticity(df, df_elast_up, df_cons_up)
    except Exception as e:
        st.error(f"Error cargando/uniendo para elasticidad: {e}")
        st.stop()

    if cat_sel != "All" and "CATEGORIA" in dj.columns:
        dj = dj[dj["CATEGORIA"] == cat_sel]
    if sub_sel != "All" and "SUBCATEGORIA" in dj.columns:
        dj = dj[dj["SUBCATEGORIA"] == sub_sel]
    dj = dj[(dj["PERIOD"] >= d_start) & (dj["PERIOD"] <= d_end)]

    if not dj.empty:
        dj = apply_ranges_block(dj)

    chart_h_local = chart_h

    def build_total_series(df_):
        d = df_.copy()
        d["UC"] = pd.to_numeric(d["UC"], errors="coerce")
        d["dln_uc"] = pd.to_numeric(d["dln_uc"], errors="coerce")
        d["dln_precio"] = pd.to_numeric(d["dln_precio"], errors="coerce")
        d["impacto_esperado"] = pd.to_numeric(d["impacto_esperado"], errors="coerce")
        if "real_clipped" in d.columns:
            d["real_clipped"] = pd.to_numeric(d["real_clipped"], errors="coerce")
        else:
            d["real_clipped"] = np.nan

        uc_by_p = d.groupby("PERIOD", dropna=False)["UC"].sum(min_count=1).sort_index()
        dln_uc_total = np.log(uc_by_p).diff()

        d = d.sort_values(["SKU", "PERIOD"]).copy()
        d["UC_lag"] = d.groupby("SKU")["UC"].shift(1)

        uc_lag_total = d.groupby("PERIOD")["UC_lag"].sum(min_count=1)
        d = d.merge(uc_lag_total.rename("UC_lag_total"), left_on="PERIOD", right_index=True, how="left")
        d["w"] = np.where(d["UC_lag_total"] > 0, d["UC_lag"] / d["UC_lag_total"], np.nan)

        exp_w = (d["w"] * d["impacto_esperado"]).groupby(d["PERIOD"]).sum(min_count=1)
        real_w = (d["w"] * d["dln_uc"]).groupby(d["PERIOD"]).sum(min_count=1)
        real_clip_w = (d["w"] * d["real_clipped"]).groupby(d["PERIOD"]).sum(min_count=1)

        out = pd.DataFrame({
            "PERIOD": uc_by_p.index,
            "UC_total": uc_by_p.values,
            "dln_uc_total": dln_uc_total.values,
            "exp_w": exp_w.reindex(uc_by_p.index).values,
            "real_w": real_w.reindex(uc_by_p.index).values,
            "real_clip_w": real_clip_w.reindex(uc_by_p.index).values,
        }).sort_values("PERIOD")

        return out

    vista = st.radio("Vista",
                     ["Impacto Real vs. Impacto Esperado", "Impacto Real Ajustado vs. Impacto Esperado", "Comparar Ambos"],
                     horizontal=True)

    if level == "SKU":
        sku_pick = sku_sel
        if sku_pick == "All":
            st.info("Selecciona un **SKU** en el sidebar para ver la elasticidad a nivel SKU.")
        else:
            dsku = dj[dj["SKU"] == sku_pick].sort_values("PERIOD").copy()
            if dsku.empty:
                st.warning("No hay datos para el SKU seleccionado con los filtros actuales.")
            else:
                real_raw_all = dsku["dln_uc"]
                real_adj_all = dsku["real_clipped"]
                exp_all      = dsku["impacto_esperado"]

                real_for_kpi = real_raw_all if vista == "Impacto Real vs. Impacto Esperado" else real_adj_all

                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("MAPE (↓ mejor)",  f"{mape(real_for_kpi, exp_all):.2f}%", "secondary", "compact")
                with c2: kpi_card("SMAPE (↓ mejor)", f"{smape(real_for_kpi, exp_all):.2f}%", "secondary", "compact")
                with c3: kpi_card("DIR (↑ mejor)",   f"{diracc(real_for_kpi, exp_all):.2f}%", "secondary", "compact")

                fig = go.Figure()
                main_series = real_raw_all if vista == "Impacto Real vs. Impacto Esperado" else real_adj_all
                main_name   = "Impacto Real (%)" if vista == "Impacto Real vs. Impacto Esperado" else "Impacto Real Ajustado (%)"
                fig.add_trace(go.Scatter(
                    x=dsku["PERIOD"], y=np.expm1(main_series)*100,
                    mode="lines", name=main_name, line=dict(width=2, color=RED_REAL)
                ))
                if vista == "Comparar Ambos":
                    fig.add_trace(go.Scatter(
                        x=dsku["PERIOD"], y=np.expm1(real_raw_all)*100,
                        mode="lines", name="Impacto Real (%)", line=dict(width=2, dash='dot', color=RED_REAL)
                    ))
                fig.add_trace(go.Scatter(
                    x=dsku["PERIOD"], y=np.expm1(exp_all)*100,
                    mode="lines", name="Impacto Esperado (%)", line=dict(width=2, dash='dash', color=BLACK_REF)
                ))
                fig.update_layout(
                    hovermode="x unified",
                    xaxis_title="Periodo",
                    yaxis_title="Cambio (%)",
                    height=chart_h_local,
                    margin=dict(l=10, r=10, t=20, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        if dj.empty:
            st.warning("No hay datos con los filtros actuales.")
        else:
            total_df = build_total_series(dj).sort_values("PERIOD").copy()

            if vista == "Impacto Real vs. Impacto Esperado":
                y_real = total_df["dln_uc_total"]
                y_exp  = total_df["exp_w"]
            elif vista == "Impacto Real Ajustado vs. Impacto Esperado":
                y_real = total_df["real_clip_w"]
                y_exp  = total_df["exp_w"]
            else:
                y_real = total_df["real_clip_w"]
                y_exp  = total_df["exp_w"]

            c1, c2, c3 = st.columns(3)
            with c1: kpi_card("MAPE (↓ mejor)",  f"{mape(y_real, y_exp):.2f}%", "secondary", "compact")
            with c2: kpi_card("SMAPE (↓ mejor)", f"{smape(y_real, y_exp):.2f}%", "secondary", "compact")
            with c3: kpi_card("DIR (↑ mejor)",   f"{diracc(y_real, y_exp):.2f}%", "secondary", "compact")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=total_df["PERIOD"], y=np.expm1(y_real)*100,
                mode="lines",
                name=("Impacto Real (%)" if vista=="Impacto Real vs. Impacto Esperado" else "Impacto Real Ajustado (%)"),
                line=dict(width=2, color=RED_REAL)
            ))
            if vista == "Comparar Ambos":
                fig.add_trace(go.Scatter(
                    x=total_df["PERIOD"], y=np.expm1(total_df["dln_uc_total"])*100,
                    mode="lines", name="Impacto Real (%)", line=dict(width=2, dash='dot', color=RED_REAL)
                ))
            fig.add_trace(go.Scatter(
                x=total_df["PERIOD"], y=np.expm1(y_exp)*100,
                mode="lines", name="Impacto Esperado (%)", line=dict(width=2, dash='dash', color=BLACK_REF)
            ))
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Periodo",
                yaxis_title="Cambio (%)",
                height=chart_h_local,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

# ===== Clasificador de elasticidad =====
def classify_elasticity(e):
    if pd.isna(e): 
        return "Sin dato"
    ae = abs(float(e))
    if math.isclose(ae, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        return "Unitario"
    return "Inelástico (<1)" if ae < 1 else "Elástico (>1)"

# ===== Color por clase =====
def class_color(cls):
    if cls == "Elástico (>1)":   return COCA_RED    
    if cls == "Unitario":        return "#1F1F1F"     
    if cls == "Inelástico (<1)": return "#666666"     
    return "#B3B3B3"                                 

# =============== TAB 3: Elasticity Classification ===============
with tab3:
    st.subheader("Elasticity Map — Clasificación por nivel")

    # 1) Construcción base
    try:
        dj5 = load_joined_for_elasticity(df, df_elast_up, df_cons_up)
    except Exception as e:
        st.error(f"Error cargando/uniendo para elasticidad: {e}")
        st.stop()

    if cat_sel != "All" and "CATEGORIA" in dj5.columns:
        dj5 = dj5[dj5["CATEGORIA"] == cat_sel]
    if sub_sel != "All" and "SUBCATEGORIA" in dj5.columns:
        dj5 = dj5[dj5["SUBCATEGORIA"] == sub_sel]
    dj5 = dj5[(dj5["PERIOD"] >= d_start) & (dj5["PERIOD"] <= d_end)]

    if dj5.empty:
        st.warning("No hay datos con los filtros actuales.")
        st.stop()

    for c in ["UC", "FINAL_PE"]:
        if c in dj5.columns:
            dj5[c] = pd.to_numeric(dj5[c], errors="coerce")

    dj5 = dj5.sort_values(["SKU", "PERIOD"]).copy()
    dj5["UC_lag"] = dj5.groupby("SKU", sort=False)["UC"].shift(1)

    if level == "Total":
        entity_col = "CATEGORIA" if "CATEGORIA" in dj5.columns else "SKU"
        nivel_txt  = "Categoría" if entity_col == "CATEGORIA" else "SKU"
    elif level == "Categoría":
        entity_col = "SUBCATEGORIA" if "SUBCATEGORIA" in dj5.columns else "SKU"
        nivel_txt  = "Subcategoría" if entity_col == "SUBCATEGORIA" else "SKU"
    else:
        entity_col = "SKU"
        nivel_txt  = "SKU"

    if entity_col not in dj5.columns:
        st.info(f"No existe la columna requerida para este nivel: {entity_col}")
        st.stop()

    agg_df = (
        dj5.groupby([entity_col], dropna=False)
           .apply(lambda g: wavg_safe(g["FINAL_PE"], g["UC_lag"]))
           .reset_index(name="Elasticidad_prom")
    )

    agg_df["Clase"] = agg_df["Elasticidad_prom"].apply(classify_elasticity)

    dj5 = apply_ranges_block(dj5)
    eps_local = 1e-8
    dj5["no_price_move"] = pd.to_numeric(dj5["dln_precio"], errors="coerce").abs() < eps_local
    dj5["is_clip"] = dj5["clip_source"].isin(["P1_MIN","P1_MAX","P2_MIN","P2_MAX"])

    qual = (
        dj5.groupby(entity_col, dropna=False)
           .agg(
               obs=("SKU","size"),
               pct_no_move=("no_price_move", lambda s: np.mean(s)*100 if len(s) else np.nan),
               pct_clip=("is_clip", lambda s: np.mean(s)*100 if len(s) else np.nan)
           )
           .reset_index()
    )

    res = agg_df.merge(qual, on=entity_col, how="left")
    res["abs_el"] = res["Elasticidad_prom"].abs()
    res = res.sort_values(["Clase","abs_el"], ascending=[True, False]).reset_index(drop=True)

    total_entidades = len(res)
    n_elast   = int((res["Clase"] == "Elástico (>1)").sum())
    n_inelast = int((res["Clase"] == "Inelástico (<1)").sum())
    n_unit    = int((res["Clase"] == "Unitario").sum())
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Entidades evaluadas", f"{total_entidades:,}", "secondary")
    with c2: kpi_card("Elásticas (>1)", f"{n_elast:,}", "secondary")
    with c3: kpi_card("Inelásticas (<1)", f"{n_inelast:,}", "secondary")
    with c4: kpi_card("Unitarias (≈1)", f"{n_unit:,}", "secondary")

    plot_df = res.copy()
    plot_df["color"] = plot_df["Clase"].apply(class_color)

    title_txt = f"Elasticidad promedio por {nivel_txt}"
    fig_map = go.Figure()
    fig_map.add_trace(
        go.Bar(
            x=plot_df[entity_col].astype(str),
            y=plot_df["Elasticidad_prom"],
            marker_color=plot_df["color"],
            text=plot_df["Clase"],
            hovertemplate=(
                f"{nivel_txt}: %{{x}}<br>"
                "Elasticidad prom.: %{y:.2f}<br>"
                "Clase: %{text}<br>"
                "Obs: %{customdata[0]:,}<br>"
                "Sin mov. precio: %{customdata[1]:.1f}%<br>"
                "Clippeadas: %{customdata[2]:.1f}%<extra></extra>"
            ),
            customdata=np.stack([
                plot_df["obs"].fillna(0).astype(int),
                plot_df["pct_no_move"].astype(float),
                plot_df["pct_clip"].astype(float)
            ], axis=-1)
        )
    )
    fig_map.update_layout(
        title=title_txt,
        xaxis_title=nivel_txt,
        yaxis_title="Elasticidad precio de la demanda (ε)",
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False
    )
    fig_map.add_hline(y= 1, line_width=2, line_dash="dot", line_color="#999999")
    fig_map.add_hline(y=-1, line_width=2, line_dash="dot", line_color="#999999")
    st.plotly_chart(fig_map, use_container_width=True)

    dist_cls = res["Clase"].value_counts(dropna=False).rename_axis("Clase").reset_index(name="count")
    fig_pie = px.pie(dist_cls, values="count", names="Clase", title="Distribución de clases")
    st.plotly_chart(fig_pie, use_container_width=True)

    out_cols = [entity_col, "Elasticidad_prom", "Clase", "obs", "pct_no_move", "pct_clip"]
    st.download_button(
        f"Descargar Elasticity Map por {nivel_txt} (CSV)",
        res[out_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"elasticity_map_por_{nivel_txt.lower()}.csv",
        type="primary"
    )

    def _fmt(x, f=".2f"):
        return "-" if (x is None or pd.isna(x)) else format(float(x), f)
    share = lambda n: (n/total_entidades*100) if total_entidades else np.nan

    st.markdown(
        f"""
    <div style="margin-top:10px; font-size:14px; line-height:1.55">
    <b>Cómo leer:</b> Cada barra muestra la elasticidad promedio ponderada por <i>UC (t-1)</i> del {nivel_txt.lower()}.
    Colores: <span style="color:{COCA_RED}">rojo</span> = elástico (|ε| &gt; 1), 
    <span style="color:#666">gris</span> = inelástico (|ε| &lt; 1), 
    <span style="color:#1F1F1F">negro</span> = unitario (|ε| ≈ 1). 
    Las líneas punteadas en ±1 marcan los umbrales teóricos.<br><br>
    <b>Resumen:</b> Evaluamos <b>{total_entidades:,}</b> {nivel_txt.lower()}(s). 
    Elásticas: <b>{n_elast:,}</b> ({_fmt(share(n_elast), '.1f')}%). 
    Inelásticas: <b>{n_inelast:,}</b> ({_fmt(share(n_inelast), '.1f')}%). 
    Unitarias: <b>{n_unit:,}</b> ({_fmt(share(n_unit), '.1f')}%).<br>
    Las métricas “Sin mov. de precio” y “Clippeadas” te orientan sobre la <i>calidad/fiabilidad</i> de la medición por entidad.
    </div>
    """,
        unsafe_allow_html=True
    )

# =============== TAB 4: Diagnostics
with tab4:
    dj2 = load_joined_for_elasticity(df, df_elast_up, df_cons_up)
    if cat_sel != "All" and "CATEGORIA" in dj2.columns:
        dj2 = dj2[dj2["CATEGORIA"] == cat_sel]
    if sub_sel != "All" and "SUBCATEGORIA" in dj2.columns:
        dj2 = dj2[dj2["SUBCATEGORIA"] == sub_sel]
    dj2 = dj2[(dj2["PERIOD"] >= d_start) & (dj2["PERIOD"] <= d_end)]
    if dj2.empty:
        st.warning("No hay datos con los filtros actuales.")
    else:
        dj2 = apply_ranges_block(dj2)

        st.subheader("Calibración: Impacto Real vs. Esperado")
        use_adjusted = st.checkbox("Usar Impacto Real Ajustado (clipped)", value=True)

        if level == "SKU" and sku_sel != "All":
            dd = dj2[dj2["SKU"] == sku_sel].sort_values("PERIOD")
            y_real = dd["real_clipped"] if use_adjusted else dd["dln_uc"]
            x_exp  = dd["impacto_esperado"]
            hover  = dd["PERIOD"].dt.strftime("%Y-%m")
        else:
            grp = dj2.groupby("PERIOD", dropna=False)
            y_real = (grp["real_clipped"].sum(min_count=1) if use_adjusted 
                      else np.log(pd.to_numeric(grp["UC"].sum(min_count=1), errors="coerce")).diff())
            x_exp  = grp["impacto_esperado"].sum(min_count=1)
            hover  = y_real.index.strftime("%Y-%m")

        x = pd.to_numeric(x_exp, errors="coerce")
        y = pd.to_numeric(y_real, errors="coerce")
        mask = x.replace([np.inf,-np.inf], np.nan).notna() & y.replace([np.inf,-np.inf], np.nan).notna()
        x, y, hover = x[mask], y[mask], np.array(list(np.array(hover)[mask]))

        if len(x) >= 2:
            b, a = np.polyfit(x, y, 1)   # y = b*x + a
            y_hat = b*x + a
            ss_res = np.sum((y - y_hat)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        else:
            b, a, r2 = np.nan, np.nan, np.nan

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("Pendiente (β)", f"{b:.2f}" if pd.notna(b) else "-", "secondary")
        with c2: kpi_card("Intercepto (α)", f"{a:.2f}" if pd.notna(a) else "-", "secondary")
        with c3: kpi_card("R²", f"{r2:.3f}" if pd.notna(r2) else "-", "secondary")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name="Periodos", marker=dict(size=7, color=RED_REAL),
            text=hover, hovertemplate="Período: %{text}<br>Esperado: %{x:.3f}<br>Real: %{y:.3f}"
        ))
        xy = np.concatenate([x.to_numpy(), y.to_numpy()])
        xy = xy[~np.isnan(xy) & ~np.isinf(xy)]
        dom = np.array([np.nanmin(xy), np.nanmax(xy)]) if xy.size else np.array([0, 1])
        fig.add_trace(go.Scatter(
            x=dom, y=dom, mode="lines", name="y=x",
            line=dict(color=BLACK_REF, width=2, dash="dash")
        ))
        if pd.notna(b) and pd.notna(a):
            fig.add_trace(go.Scatter(
                x=dom, y=b*dom + a, mode="lines", name="Ajuste OLS",
                line=dict(color=RED_REAL, width=2)
            ))
        fig.update_layout(
            xaxis_title="Impacto Esperado (Δln UC)",
            yaxis_title=("Impacto Real Ajustado" if use_adjusted else "Impacto Real") + " (Δln UC)",
            height=420, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
        # --- Texto dinámico de interpretación (debajo del scatter) ---
        fmt = lambda x, f: "-" if (x is None or pd.isna(x)) else format(float(x), f)

        def calidad_r2(v):
            if pd.isna(v): return "sin poder explicativo (R² no disponible)"
            if v < 0.50:  return "baja capacidad explicativa"
            if v < 0.75:  return "capacidad explicativa media"
            return "buena capacidad explicativa"

        def lectura_beta(v):
            if pd.isna(v): return "pendiente no disponible."
            if v < 0:     return "relación inversa (signo contrario)."
            if v < 1:     return "sub-respuesta (el real se mueve menos que lo esperado)."
            if v == 1:    return "respuesta 1:1 (calibración ideal)."
            return "sobre-respuesta (el real se mueve más que lo esperado)."

        real_label = "Impacto Real Ajustado (clipped)" if use_adjusted else "Impacto Real"
        beta_txt  = fmt(b, ".2f")
        alpha_txt = fmt(a, ".2f")
        r2_txt    = fmt(r2, ".3f")

        st.markdown(
            f"""
        <div style="margin-top:8px; font-size:14px; line-height:1.55">
        Con <i>{real_label}</i> como variable dependiente y el impacto esperado como explicativa,
        la <b>pendiente β</b> es <b>{beta_txt}</b>, lo que sugiere {lectura_beta(b)}
        El <b>intercepto α</b> es <b>{alpha_txt}</b> (sesgo constante; ideal ≈ 0).
        El <b>R²</b> es <b>{r2_txt}</b>, indicando {calidad_r2(r2)}.
        </div>
        """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.subheader("Clipping dashboard")

        is_clip = dj2["clip_source"].isin(["P1_MIN","P1_MAX","P2_MIN","P2_MAX"])
        clip_rate = np.mean(is_clip) * 100 if len(dj2) else np.nan
        with st.container():
            c1, c2 = st.columns(2)
            with c1: kpi_card("% Observaciones clippeadas", f"{clip_rate:.1f}%" if pd.notna(clip_rate) else "-", "secondary")
            no_move_rate = np.mean(dj2["clip_source"] == "NO_PRICE_MOVE") * 100 if len(dj2) else np.nan
            with c2: kpi_card("% Sin movimiento de precio", f"{no_move_rate:.1f}%" if pd.notna(no_move_rate) else "-", "secondary")

        dist = (dj2["clip_source"].value_counts(dropna=False)
                .rename_axis("clip_source").reset_index(name="count"))
        dist["clip_source"] = dist["clip_source"].astype(str)
        fig_bar = px.bar(dist, x="clip_source", y="count", title="Distribución de clip_source")
        fig_bar.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ====== Narrativa dinámica para "Distribución de clip_source" ======

        # Totales y KPIs (usa las mismas variables ya calculadas arriba)
        total_obs   = len(dj2)
        n_clip      = int(is_clip.sum()) if total_obs else 0
        pct_clip    = (n_clip / total_obs * 100) if total_obs else np.nan
        n_no_move   = int((dj2["clip_source"] == "NO_PRICE_MOVE").sum()) if total_obs else 0
        pct_no_move = (n_no_move / total_obs * 100) if total_obs else np.nan

        # Dist normalizada (usa el 'dist' que ya creaste para la gráfica)
        dist_norm = dist.copy()
        dist_norm["pct"] = np.where(total_obs > 0, dist_norm["count"] / total_obs * 100, np.nan)
        dist_norm = dist_norm.sort_values("count", ascending=False)

        # Descripciones por categoría
        desc_map = {
            "None": "sin clipping; la elasticidad observada quedó dentro de los rangos válidos (P1/P2).",
            "P1_MIN": "elasticidad < mínimo del rango P1; se elevó al límite inferior P1.",
            "P1_MAX": "elasticidad > máximo del rango P1; se redujo al límite superior P1.",
            "P2_MIN": "elasticidad < mínimo del rango P2; se elevó al mínimo P2.",
            "P2_MAX": "elasticidad > máximo del rango P2; se redujo al máximo P2.",
            "NO_PRICE_MOVE": "sin cambio de precio (Δln Precio ≈ 0); el impacto real se fija en 0."
        }

        # Helper de formato
        fmt_pct  = lambda v: "-" if (v is None or pd.isna(v)) else f"{v:,.1f}%"
        fmt_int  = lambda n: "-" if (n is None or pd.isna(n)) else f"{int(n):,}"

        # Frase de resumen (fuente principal de clipping)
        top_clip = (dist_norm[~dist_norm["clip_source"].isin(["None", "NO_PRICE_MOVE"])]
                    .head(1))
        if len(top_clip):
            top_name  = str(top_clip.iloc[0]["clip_source"])
            top_count = int(top_clip.iloc[0]["count"])
            top_pct   = float(top_clip.iloc[0]["pct"])
            resumen_fuente = f"La principal fuente de clipping fue <b>{top_name}</b> ({fmt_int(top_count)} obs; {fmt_pct(top_pct)})."
        else:
            resumen_fuente = "No se detectó una fuente dominante de clipping (P1/P2)."

        # Construye lista por categoría presente
        items = []
        for _, row in dist_norm.iterrows():
            name  = str(row["clip_source"])
            cnt   = int(row["count"])
            pct   = float(row["pct"]) if not pd.isna(row["pct"]) else np.nan
            desc  = desc_map.get(name, "categoría no documentada.")
            items.append(
                f"<li><b>{name}</b>: {desc} "
                f"<span style='color:#666'>({fmt_int(cnt)} obs; {fmt_pct(pct)})</span></li>"
            )
        ul_html = "<ul style='margin:6px 0 0 18px;'>" + "\n".join(items) + "</ul>"

        # Render del párrafo explicativo
        st.markdown(
            f"""
        <div style="margin-top:8px; font-size:14px; line-height:1.55">
        <b>Lectura de la distribución:</b> Se analizaron <b>{fmt_int(total_obs)}</b> observaciones.
        De ellas, <b>{fmt_pct(pct_clip)}</b> ({fmt_int(n_clip)}) fueron <i>clippeadas</i> (KPI “% Observaciones clippeadas”)
        y <b>{fmt_pct(pct_no_move)}</b> ({fmt_int(n_no_move)}) no tuvieron movimiento de precio
        (KPI “% Sin movimiento de precio”). {resumen_fuente}
        <br><br>
        <b>Detalle por barra:</b>
        {ul_html}
        </div>
        """,
            unsafe_allow_html=True,
        )


        # sku_clip = (dj2.assign(is_clip=is_clip)
        #                .groupby("SKU")["is_clip"].mean().sort_values(ascending=False)
        #                .head(15).reset_index())
        # sku_clip["is_clip"] = sku_clip["is_clip"] * 100
        # fig_rank = px.bar(sku_clip, x="SKU", y="is_clip", title="Top SKUs por % de clipping")
        # fig_rank.update_layout(yaxis_title="% clip", height=360, margin=dict(l=10,r=10,t=40,b=10))
        # st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("---")
        st.subheader("Waterfall del GAP (Real – Esperado) por SKU")

        gap_view = st.radio("Usar real", ["Ajustado (clipped)", "Sin ajustar"], horizontal=True)
        use_adj_wf = (gap_view == "Ajustado (clipped)")
        per_opts = sorted(dj2["PERIOD"].dropna().unique())
        per_pick = st.selectbox("Periodo", options=per_opts, index=len(per_opts)-1, format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m"))
        n_top = st.slider("Top-N SKUs", 5, 30, 15)

        dper = dj2[dj2["PERIOD"] == per_pick].copy()
        real_col = "real_clipped" if use_adj_wf else "dln_uc"
        dper["gap"] = dper[real_col] - dper["impacto_esperado"]

        contrib = (dper.groupby("SKU")["gap"].sum()
                        .reindex(dper["SKU"].unique())
                        .fillna(0.0))
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).head(n_top).index)
        others = (dper["gap"].sum() - contrib.sum())

        labels = list(contrib.index) + (["Otros"] if abs(others) > 1e-12 else [])
        values = list(contrib.values) + ([others] if abs(others) > 1e-12 else [])
        measures = ["relative"] * len(values)

        wf = go.Figure(go.Waterfall(
            name="gap",
            orientation="v",
            x=labels,
            measure=measures,
            y=np.array(values) * 100,
            connector={"line":{"color":"#BBBBBB"}},
            increasing={"marker":{"color":"#1F1F1F"}},
            decreasing={"marker":{"color":RED_REAL}},
        ))
        wf.update_layout(
            title=f"GAP Real {'Ajustado' if use_adj_wf else 'Sin ajustar'} – Esperado · {pd.Timestamp(per_pick).strftime('%Y-%m')}",
            yaxis_title="pp (puntos porcentuales)",
            height=420, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(wf, use_container_width=True)

        # ===== Narrativa dinámica para Waterfall GAP =====
        fmt_pp   = lambda v: "-" if v is None or pd.isna(v) else f"{v:,.1f} pp"
        fmt_int  = lambda n: "-" if n is None or pd.isna(n) else f"{int(n):,}"
        fmt_nm   = lambda s: "-" if s is None else str(s)

        period_txt = pd.Timestamp(per_pick).strftime("%Y-%m")
        real_label = "Ajustado (clipped)" if use_adj_wf else "Sin ajustar"

        # GAP total (en pp)
        gap_total_pp = float(dper["gap"].sum() * 100)

        # Desglose Top positivos/negativos entre los seleccionados para el Waterfall
        contrib_sorted = contrib.sort_values(ascending=False)
        top_pos = contrib_sorted[contrib_sorted > 0].head(3)   # mayores aportes positivos
        top_neg = contrib_sorted[contrib_sorted < 0].tail(3)   # más negativos (cola)

        otros_pp = others * 100 if abs(others) > 1e-12 else 0.0

        def list_items(series):
            items = []
            for sk, v in series.items():
                items.append(f"<li><b>{sk}</b>: {fmt_pp(v*100)}</li>")
            if not items:
                items = ["<li>No hay contribuciones en este grupo.</li>"]
            return "<ul style='margin:6px 0 0 18px'>" + "\n".join(items) + "</ul>"

        ul_pos = list_items(top_pos)
        ul_neg = list_items(top_neg)

        signo = "positivo (Real > Esperado)" if gap_total_pp > 0 else ("negativo (Real < Esperado)" if gap_total_pp < 0 else "nulo")

        st.markdown(
        f"""<div style="margin-top:8px; font-size:14px; line-height:1.55">
        <b>Cómo leer esta gráfica:</b> cada barra muestra la contribución de un SKU al <i>GAP</i> = Real − Esperado
        (en <b>puntos porcentuales</b>).<br>
        Barras <span style="color:#000">negras</span> = contribuciones <b>positivas</b> (el real superó lo esperado).
        Barras <span style="color:{RED_REAL}">rojas</span> = contribuciones <b>negativas</b> (el real quedó debajo de lo esperado).
        El bucket <b>“Otros”</b> acumula todos los SKUs fuera del Top-N seleccionado.

<br><br>
<b>Contexto de la vista:</b> Periodo <b>{period_txt}</b> · Real usado: <b>{real_label}</b> · Top-N SKUs: <b>{fmt_int(n_top)}</b>.<br>
<b>GAP total</b>: <b>{fmt_pp(gap_total_pp)}</b>, de signo <b>{signo}</b>.

<b>Principales aportes positivos</b> (impulsan el GAP hacia arriba):
{ul_pos}

<b>Principales aportes negativos</b> (empujan el GAP hacia abajo):
{ul_neg}

<b>Otros</b>: {fmt_pp(otros_pp)} (suma del resto de SKUs fuera del Top-N).
</div>""",
        unsafe_allow_html=True
        )



# =============== TAB 5: Revenue Decomposition
with tab5:
    st.subheader("Descomposición de ΔIngresos: Precio · Volumen · Mix")
    d_rev = df.copy()
    if cat_sel != "All" and "CATEGORIA" in d_rev.columns:  d_rev = d_rev[d_rev["CATEGORIA"] == cat_sel]
    if sub_sel != "All" and "SUBCATEGORIA" in d_rev.columns: d_rev = d_rev[d_rev["SUBCATEGORIA"] == sub_sel]
    if sku_sel != "All" and "SKU" in d_rev.columns:          d_rev = d_rev[d_rev["SKU"] == sku_sel]
    d_rev = d_rev[(d_rev["PERIOD"] >= d_start) & (d_rev["PERIOD"] <= d_end)]

    if not {"INGRESO","UC"}.issubset(set(d_rev.columns)):
        st.info("Se requieren columnas INGRESO y UC para la descomposición.")
    else:
        d_rev = d_rev.sort_values(["SKU","PERIOD"]).copy()
        d_rev["P"] = np.where(pd.to_numeric(d_rev["UC"], errors="coerce")>0,
                              pd.to_numeric(d_rev["INGRESO"], errors="coerce")/pd.to_numeric(d_rev["UC"], errors="coerce"),
                              np.nan)
        d_rev["P_lag"] = d_rev.groupby("SKU")["P"].shift(1)
        d_rev["Q_lag"] = d_rev.groupby("SKU")["UC"].shift(1)
        d_rev["P_diff"] = d_rev["P"] - d_rev["P_lag"]
        d_rev["Q_diff"] = d_rev["UC"] - d_rev["Q_lag"]

        df_ben = (d_rev
                  .assign(price_eff = d_rev["Q_lag"] * d_rev["P_diff"],
                          vol_eff   = d_rev["P_lag"] * d_rev["Q_diff"],
                          mix_eff   = d_rev["P_diff"] * d_rev["Q_diff"])
                  .groupby("PERIOD", dropna=False)[["price_eff","vol_eff","mix_eff"]]
                  .sum(min_count=1)
                  .reset_index()
                  .sort_values("PERIOD"))

        df_long = df_ben.melt(id_vars="PERIOD", var_name="Componente", value_name="ΔIngresos")
        names_map = {"price_eff":"Precio", "vol_eff":"Volumen", "mix_eff":"Mix"}
        df_long["Componente"] = df_long["Componente"].map(names_map)

        fig_dec = px.bar(df_long, x="PERIOD", y="ΔIngresos", color="Componente", barmode="relative",
                         title="ΔIngresos por componente (Bennett)")
        fig_dec.update_layout(
            height=440, margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(orientation="h")
        )
        fig_dec.add_hline(
            y=0,
            line_width=3,
            line_dash="solid",
            line_color="black"
        )
        st.plotly_chart(fig_dec, use_container_width=True)

        st.download_button(
            "Descargar descomposición (CSV)",
            df_long.to_csv(index=False).encode("utf-8"),
            file_name="revenue_decomposition_bennett.csv",
            type="primary"
        )
