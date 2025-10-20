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
    
    if "SKU" in d.columns:
        d["SKU"] = (
            d["SKU"]
            .astype(str)
            .str.upper()
            .str.replace(r"\bLATA\b", "ALUMINIO", regex=True)
            .str.replace(r"\bLATAS\b", "ALUMINIO", regex=True)
        )

    for c in ["CATEGORIA", "SUBCATEGORIA", "MARCA"]:
        if c in d.columns:
            d[c] = _norm_series(d[c])
    return d

@reactive.Calc
def df_vtm_map() -> pd.DataFrame:
    """
    Mapea los flujos de volumen (VTM) con jerarquía consistente.
    - Usa SOURCE_SKU para derivar la Marca (no EMBOTELLADOR_REVISED)
    - Añade SUBCATEGORIA desde mt_consolidated_pe
    - Marca banderas de fabricante Coca-Cola en origen y destino
    """
    try:
        vtm = read_table_local("udm_vtm_melted_mapping")
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "SOURCE_SKU","DESTINATION_SKU","SRC_MFG","DEST_MFG",
            "VOLUME_TRANSFER","UNIT_MIX","REVISED_CATEGORY"
        ])

    # --- Normalización de nombres base
    vtm = vtm.rename(columns={
        "REVISED_CATEGORY": "CATEGORIA",
        "VOLUME_TRANSFER": "VOLUME_TRANSFER"
    })

    # --- Asegurar columnas requeridas
    for col in ["SOURCE_SKU", "DESTINATION_SKU", "SRC_MFG", "DEST_MFG"]:
        if col not in vtm.columns:
            vtm[col] = np.nan

    # --- Derivar Marca desde SOURCE_SKU
    import re

    def extraer_marca(source_sku: str) -> str:
        """
        Extrae la 'marca' del texto SOURCE_SKU tomando todo lo que aparece
        antes de las palabras clave PET, VIDRIO, TETRAPACK, LATA, GARRAFON o REF.
        Si no hay coincidencia, devuelve las primeras 1–2 palabras.
        """
        if not isinstance(source_sku, str) or not source_sku.strip():
            return "DESCONOCIDO"
        s = source_sku.upper().strip()
        s = s.replace("-", " ").replace("_", " ").replace(".", " ")
        patron = r"^(.*?)(?=\b(PET|VIDRIO|TETRAPACK|LATA|GARRAFON|REF)\b)"
        m = re.search(patron, s)
        if m:
            marca = m.group(1).strip()
        else:
            partes = s.split()
            marca = " ".join(partes[:2]).strip()

        # --- Limpieza de caracteres ---
        marca = re.sub(r"[^A-Z0-9ÁÉÍÓÚÑ ]+", "", marca).strip()

        # --- Corrección para mantener formato con guiones ---
        marca = re.sub(r"\bCOCA COLA\b", "COCA-COLA", marca)
        marca = re.sub(r"\bSIN AZUCAR\b", "SIN AZUCAR", marca)
        return marca or "DESCONOCIDO"

    vtm["MARCA"] = vtm["SOURCE_SKU"].apply(extraer_marca)

    # --- Traer SUBCATEGORÍA desde mt_consolidated_pe
    try:
        mt = read_table_local("mt_consolidated_pe")[["PPG", "SUB_CATEGORY"]]
        mt = mt.rename(columns={"PPG": "SOURCE_SKU", "SUB_CATEGORY": "SUBCATEGORIA"})
        vtm = vtm.merge(mt, on="SOURCE_SKU", how="left")
    except FileNotFoundError:
        vtm["SUBCATEGORIA"] = np.nan

    # --- Normalizar claves y texto
    for c in ["CATEGORIA", "SUBCATEGORIA", "MARCA", "SOURCE_SKU", "DESTINATION_SKU"]:
        vtm[c] = vtm[c].astype(str).str.strip().str.upper()

    # --- Banderas de fabricante
    vtm["SRC_MFG_KEY"] = _norm_series(vtm["SRC_MFG"])
    vtm["DEST_MFG_KEY"] = _norm_series(vtm["DEST_MFG"])
    vtm["IS_SRC_COCA"] = vtm["SRC_MFG_KEY"].str.contains("COCA", na=False)
    vtm["IS_DEST_COCA"] = vtm["DEST_MFG_KEY"].str.contains("COCA", na=False)

    # --- Filtrar registros con VOLUME_TRANSFER válido
    vtm["VOLUME_TRANSFER"] = pd.to_numeric(vtm["VOLUME_TRANSFER"], errors="coerce").fillna(0)
    vtm = vtm[vtm["VOLUME_TRANSFER"] > 0].copy()

    return vtm


def last_full_year_window(dseries: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (start, end_exclusive) del último año completo según PERIOD mensual disponible."""
    if dseries.empty:
        today = pd.Timestamp.today().normalize()
        end_m = pd.Timestamp(year=today.year, month=today.month, day=1)
    else:
        end_m = pd.to_datetime(dseries["PERIOD"]).max()
        end_m = pd.Timestamp(year=end_m.year, month=end_m.month, day=1)
    start_m = (end_m - pd.offsets.DateOffset(months=24)).normalize()
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
    html, body {
    font-family: 'Montserrat', sans-serif;
    height: 100%;
    overflow-y: auto;
    }

    /* ---- Navegación de tabs ---- */
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

    /* ---- Sidebar que se desplaza junto con el scroll ---- */
    .bslib-page-sidebar {
        display: flex !important;        
        align-items: flex-start !important;
        overflow: visible !important;   
    }

    .sidebar-scroll {
        position: sticky !important;
        top: 10px;
        align-self: flex-start;
        max-height: calc(100vh - 20px);
        overflow-y: auto;
        background-color: #ffffff;
        padding: 15px;
        border-right: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        scrollbar-width: thin;
        scrollbar-color: #E41E26 #f5f5f5;
    }

    /* Scrollbar personalizado (para Chrome, Edge, Safari) */
    .sidebar-scroll::-webkit-scrollbar {
        width: 8px;
    }
    .sidebar-scroll::-webkit-scrollbar-thumb {
        background-color: #E41E26;
        border-radius: 4px;
    }
    .sidebar-scroll::-webkit-scrollbar-track {
        background: #f5f5f5;
    }

    /* ---- Alineación del layout ---- */
    .layout-sidebar {
        display: flex;
        align-items: flex-start !important;
        overflow: visible !important;
    }

    /* ---- Contenido principal ---- */
    .main-content {
    flex: 1;
    overflow: visible;
    padding: 20px;
    }

    /* ---- Tipografía secundaria ---- */
    .muted {
    color: #666;
    font-size: 12px;
    }
    """),

    ui.tags.style("""
    /* === Fila de gráficos lado a lado === */
    .fila-graficos {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 20px;
        flex-wrap: wrap;  /* permite que se apilen en pantallas pequeñas */
        width: 100%;
        margin: 30px auto;
        max-width: 95vw;
    }

    .grafico-izquierdo, .grafico-derecho {
        flex: 1 1 48%;
        max-width: 48%;
        min-width: 400px;
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    .grafico-izquierdo .plotly-container,
    .grafico-derecho .plotly-container {
        width: 100% !important;
        height: auto !important;
    }

    @media (max-width: 1100px) {
        .grafico-izquierdo, .grafico-derecho {
            flex: 1 1 100%;
            max-width: 100%;
        }
    }
    .bloque-elasticidad {
        width: 100%;
        max-width: 95vw;
        margin: 25px auto;
        background-color: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
        padding: 15px 25px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    """),
    # --- CSS para el estilo y posición del header ---
    ui.tags.style("""
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            padding: 0px;
        }

        .dashboard-title {
            font-size: 2rem;
            font-weight: 700;
            color: #2E2E2E;
            font-family: 'Montserrat', sans-serif;
        }

        .dashboard-logo {
            height: 170px;
            object-fit: contain;
        }
    """),

    # --- Encabezado principal ---
    ui.div(
        {"class": "dashboard-header"},
        ui.h1("Monitor de Elasticidad de Precio", class_="dashboard-title"),
        ui.img(src="logo.png", class_="dashboard-logo")
    ),

    ui.page_sidebar(
        ui.sidebar(
            ui.output_ui("cat_sel_ui"),
            ui.output_ui("sub_sel_ui"),
            ui.output_ui("marca_sel_ui"),
            ui.output_ui("sku_sel_ui"),
            ui.output_ui("date_range_ui"),
            width=200,
            class_="sidebar-scroll"
        ),

        ui.navset_tab(
            # ---- NUEVA pestaña unificada: Sensibilidad de Precio y Volumen
            ui.nav_panel(
                ui.tags.span("Sensibilidad de", ui.br(), "Precio y Volumen"),
                
                # === CONTENEDOR PRINCIPAL CON MARGENES ===
                ui.div(
                    # --- KPIs Superiores ---
                    ui.div(
                        ui.row(
                            ui.column(3, ui.output_ui("kpi_riesgo_externo")),
                            ui.column(3, ui.output_ui("kpi_demanda_perdida")),
                            ui.column(3, ui.output_ui("kpi_canibal_total")),
                            ui.column(3, ui.output_ui("kpi_indice_riesgo_global")),
                        ),
                        style=(
                            "margin-top: 25px; "
                            "margin-bottom: 35px; "
                            "padding-top: 15px; "
                            "padding-bottom: 15px; "
                            "border-top: 2px solid #E9ECEF; "
                            "border-bottom: 2px solid #E9ECEF;"
                        )
                    ),

                    # --- Tabla de sensibilidad ---
                    ui.div(
                        ui.h5("Análisis de Sensibilidad de Precio y Volumen", 
                            style="text-align:center; font-weight:600; margin-bottom:20px; color:#4B4B4B;"),
                        ui.output_data_frame("tabla_sensibilidad"),
                        style=(
                            "max-width: 90%; "
                            "margin: 0 auto; "
                            "margin-top: 10px; "
                            "margin-bottom: 40px;"
                        )
                    ),

                    # --- Diagrama Sankey ---
                    ui.div(
                        ui.h5("Flujos de Volumen", 
                            style="text-align:center; font-weight:600; margin-bottom:15px; color:#4B4B4B;"),
                        output_widget("fig_sankey"),
                        style=(
                            "max-width: 90%; "
                            "margin: 0 auto; "
                            "margin-top: 40px; "
                            "margin-bottom: 60px;"
                        )
                    ),

                    # --- Botón de descarga centrado ---
                    ui.div(
                        ui.download_button("dl_sensibilidad", "Descargar análisis (CSV)"),
                        style="text-align:center; margin-top:20px; margin-bottom:40px;"
                    ),
                )
            ),

            # ---- Línea de tiempo
            ui.nav_panel(
                ui.tags.span("Línea de tiempo", ui.br(), "de Elasticidad"),
                ui.div(
                    ui.row(
                        ui.column(3, ui.output_ui("kpi_1")),
                        ui.column(3, ui.output_ui("kpi_4")), 
                        ui.column(3, ui.output_ui("kpi_2")),
                        ui.column(3, ui.output_ui("kpi_3")),
                    ),
                    style="display:flex; justify-content:center; align-items:stretch; gap:10px; margin:10px 0;"
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

                # === CONTENEDOR FLEX PADRE ===
                ui.div(
                    {"class": "fila-graficos"},
                    ui.div(
                        {"class": "grafico-izquierdo"},
                        output_widget("fig_mapa")
                    ),
                    ui.div(
                        {"class": "grafico-derecho"},
                        output_widget("fig_pie")
                    ),
                    ui.div(
                        {"class": "bloque-elasticidad"},
                        ui.HTML("""
                        <p style="font-size:15px; color:#2E2E2E; margin-bottom:6px;">
                            <b>Interpretación general de elasticidad de precios</b>
                        </p>
                        <p style="font-size:13px; color:#4B4B4B; line-height:1.5;">
                            <span style="color:#E41E26; font-weight:600;">Elástico (&gt;1):</span> 
                            Demanda altamente sensible al precio.<br>
                            <span style="color:#4B4B4B; font-weight:600;">Inelástico (&lt;1):</span> 
                            Demanda estable con respecto al precio.<br>
                            <span style="color:#000000; font-weight:600;">Unitario (=1):</span> 
                            Relación proporcional entre precio y volumen.
                        </p>
                        """)
                    ),
                ),

                ui.div(
                    ui.download_button("dl_mapa", "Descargar mapa de elasticidad (CSV)"),
                    style="text-align:center; margin-top:10px;"
                )
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
def infer_level(cat_sel, sub_sel, marca_sel, sku_sel):
    if sku_sel != "All":
        return "SKU"
    elif marca_sel != "All":
        return "Marca"
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
        return infer_level(input.cat_sel() or "All", input.sub_sel() or "All", input.marca_sel() or "All", input.sku_sel() or "All")
    
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
        mt2["SKU_KEY"] = _norm_series(mt2["SKU"])
        w["SKU_KEY"] = _norm_series(w["SKU"])
        j = mt2.merge(w[["SKU_KEY", "W_CU"]], on="SKU_KEY", how="left")
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
        if dims["sku"] != "All" and "SKU_KEY" in j.columns:
            row = j.loc[j["SKU_KEY"] == _norm_val(dims["sku"])].head(1)
            if len(row):
                if "CATEGORIA_KEY" in row.columns:    dims_key["cat"]   = row["CATEGORIA_KEY"].iloc[0]
                if "SUBCATEGORIA_KEY" in row.columns: dims_key["sub"]   = row["SUBCATEGORIA_KEY"].iloc[0]
                if "MARCA_KEY" in row.columns:        dims_key["marca"] = row["MARCA_KEY"].iloc[0]
                dims_key["sku"] = row["SKU_KEY"].iloc[0]
        return dims_key

    def _mask_level(j: pd.DataFrame, dims_key: dict) -> pd.Series:
        # Aplica todos los filtros activos (cat/sub/marca/sku) de forma acumulativa
        m = pd.Series(True, index=j.index)
        if "CATEGORIA_KEY" in j.columns and dims_key["cat"]   != "ALL": m &= (j["CATEGORIA_KEY"]   == dims_key["cat"])
        if "SUBCATEGORIA_KEY" in j.columns and dims_key["sub"]!= "ALL": m &= (j["SUBCATEGORIA_KEY"]== dims_key["sub"])
        if "MARCA_KEY" in j.columns and dims_key["marca"]     != "ALL": m &= (j["MARCA_KEY"]       == dims_key["marca"])
        if "SKU_KEY" in j.columns and dims_key["sku"]         != "ALL": m &= (j["SKU_KEY"]         == dims_key["sku"])
        return m

    def _wavg_metric_at_level(j: pd.DataFrame, metric: str, f: dict):
        if j.empty or metric not in j.columns:
            return np.nan
        dims_key = _effective_dims(j, f)
        sel = _mask_level(j, dims_key)
        vv = pd.to_numeric(j.loc[sel, metric], errors="coerce")
        ww = pd.to_numeric(j.loc[sel, "W_CU"], errors="coerce").clip(lower=0)
        mask = vv.notna() & ww.notna() & (ww > 0)
        if not mask.any():
            return np.nan
        return float(np.average(vv[mask], weights=ww[mask]))
    
    def vtm_percentages_from_mapping(df_map: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula para cada SKU origen (SOURCE_SKU) los porcentajes de volumen transferido
        hacia productos internos (Coca-Cola), hacia competencia y la demanda perdida.
        """
        d = df_map.copy()
        if d.empty:
            return pd.DataFrame(columns=[
                "SOURCE_SKU_KEY", "Pct_interno", "Pct_competencia", "Pct_demanda_perdida"
            ])
        
        # --- Normalizar columnas clave ---
        for col in ["SOURCE_SKU", "DESTINATION_SKU", "SRC_MFG", "DEST_MFG"]:
            if col not in d.columns:
                raise KeyError(f"Falta la columna requerida: {col}")
        d["SOURCE_SKU_KEY"] = _norm_series(d["SOURCE_SKU"])
        d["DEST_SKU_KEY"] = _norm_series(d["DESTINATION_SKU"])
        d["SRC_MFG_KEY"] = _norm_series(d["SRC_MFG"])
        d["DEST_MFG_KEY"] = _norm_series(d["DEST_MFG"])

        # --- Identificar tipo de flujo ---
        d["IS_SRC_COCA"] = d["SRC_MFG_KEY"].str.contains("COCA", na=False)
        d["IS_DEST_COCA"] = d["DEST_MFG_KEY"].str.contains("COCA", na=False)
        d["VOLUME_TRANSFER"] = pd.to_numeric(d["VOLUME_TRANSFER"], errors="coerce").fillna(0)

        # --- Agregaciones por SKU origen ---
        agg = (
            d[d["IS_SRC_COCA"]]  # solo fuentes Coca-Cola
            .groupby("SOURCE_SKU_KEY", dropna=False)
            .apply(lambda g: pd.Series({
                "Vol_total": g["VOLUME_TRANSFER"].sum(),
                "Vol_interno": g.loc[g["IS_DEST_COCA"], "VOLUME_TRANSFER"].sum(),
                "Vol_competencia": g.loc[~g["IS_DEST_COCA"], "VOLUME_TRANSFER"].sum()
            }))
            .reset_index()
        )

        # --- Calcular proporciones ---
        agg["Vol_total"] = agg["Vol_total"].replace(0, np.nan)
        agg["Pct_interno"] = (agg["Vol_interno"] / agg["Vol_total"]).clip(0, 1)
        agg["Pct_competencia"] = (agg["Vol_competencia"] / agg["Vol_total"]).clip(0, 1)
        agg["Pct_demanda_perdida"] = (1 - (agg["Pct_interno"] + agg["Pct_competencia"])).clip(0, 1)

        return agg[[
            "SOURCE_SKU_KEY", "Pct_interno", "Pct_competencia", "Pct_demanda_perdida"
        ]]


    # ================= NUEVA TAB: Sensibilidad dinámica por nivel =================
    @reactive.Calc
    def sensibilidad_nivel_tabla_df():
        try:
            vm = df_vtm_map().copy()
            mt = df_mt_pe().copy()

            if vm.empty:
                print("⚠️ df_vtm_map vacío.")
                return pd.DataFrame()

            # --- Filtrar solo SKUs Coca-Cola (fabricante origen)
            vm["SRC_MFG_KEY"] = _norm_series(vm.get("SRC_MFG", pd.Series(index=vm.index, dtype=str)))
            mask_coca = vm["SRC_MFG_KEY"].str.contains("COCA COLA FABRICANTE", na=False)
            vm = vm[mask_coca].copy()
            if vm.empty:
                print("⚠️ No hay flujos con fabricante COCA.")
                return pd.DataFrame()

            # --- Limpieza básica
            vm["VOLUME_TRANSFER"] = pd.to_numeric(vm["VOLUME_TRANSFER"], errors="coerce").fillna(0)
            vm = vm[vm["VOLUME_TRANSFER"] > 0].copy()

            # --- Claves normalizadas
            vm["SOURCE_SKU_KEY"] = _norm_series(vm["SOURCE_SKU"])
            vm["DEST_MFG_KEY"] = _norm_series(vm["DEST_MFG"])
            vm["IS_DEST_COCA"] = vm["DEST_MFG_KEY"].str.contains("COCA", na=False)

            # --- Agregación base por SKU
            g = (
                vm.groupby("SOURCE_SKU_KEY", dropna=False)
                .apply(lambda s: pd.Series({
                    "vol_int": s.loc[s["IS_DEST_COCA"], "VOLUME_TRANSFER"].sum(),
                    "vol_comp": s.loc[~s["IS_DEST_COCA"], "VOLUME_TRANSFER"].sum(),
                    "cat": s["CATEGORIA"].iloc[0] if "CATEGORIA" in s.columns else "Desconocido",
                    "subcat": s["SUBCATEGORIA"].iloc[0] if "SUBCATEGORIA" in s.columns else "Desconocido",
                    "marca": s["MARCA"].iloc[0],   # ✅ usar la marca derivada
                    "sku": s["SOURCE_SKU"].iloc[0],
                }))
                .reset_index()
            )

            g["vol_map"] = g["vol_int"] + g["vol_comp"]
            g = g[g["vol_map"] > 0].copy()
            if g.empty:
                print("⚠️ No hay SKUs con volumen mapeado válido.")
                return pd.DataFrame()

            # --- Porcentajes
            g["pct_int"] = (g["vol_int"] / g["vol_map"]) * 100
            g["pct_comp"] = (g["vol_comp"] / g["vol_map"]) * 100
            g["pct_loss"] = (100 - (g["pct_int"] + g["pct_comp"])).clip(lower=0)

            # --- Merge con elasticidades
            mt["SKU_KEY"] = _norm_series(mt["SKU"])
            g = g.merge(
                mt[["SKU_KEY", "PPG_PE", "NET_PE", "SUBCATEGORIA", "CATEGORIA"]],
                left_on="SOURCE_SKU_KEY", right_on="SKU_KEY", how="left"
            )

            # --- Filtros jerárquicos reales
            f = filters()

            if f["cat"] != "All" and "CATEGORIA" in g.columns:
                g = g[_norm_series(g["CATEGORIA"]) == _norm_val(f["cat"])]

            if f["sub"] != "All" and "SUBCATEGORIA" in g.columns:
                g = g[_norm_series(g["SUBCATEGORIA"]) == _norm_val(f["sub"])]

            if f["marca"] != "All" and "marca" in g.columns:
                marca_filtro = _norm_val(f["marca"]).replace("-", " ")
                g = g[_norm_series(g["marca"]).str.replace("-", " ") == marca_filtro]

            if f["sku"] != "All" and "sku" in g.columns:
                g = g[_norm_series(g["sku"]) == _norm_val(f["sku"])]

            if g.empty:
                print("⚠️ No hay datos después de aplicar filtros.")
                return pd.DataFrame()

            # --- Nivel jerárquico dinámico (drilldown)
            if f["sku"] != "All":
                level_col = "sku"
            elif f["marca"] != "All":
                level_col = "sku"
            elif f["sub"] != "All":
                level_col = "marca"
            elif f["cat"] != "All":
                level_col = "subcat"
            else:
                level_col = "cat"

            print(f"[DEBUG] Nivel detectado: {level_col}, Registros: {len(g)}")

            # --- Agregación ponderada
            def wavg(v, w):
                v = pd.to_numeric(v, errors="coerce")
                w = pd.to_numeric(w, errors="coerce")
                m = v.notna() & w.notna() & (w > 0)
                return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

            agg = (
                g.groupby(level_col, dropna=False)
                .apply(lambda d: pd.Series({
                    "Elasticidad Bruta": wavg(d["PPG_PE"], d["vol_map"]),
                    "Elasticidad Neta": wavg(d["NET_PE"], d["vol_map"]),
                    "% Interno": wavg(d["pct_int"], d["vol_map"]),
                    "% Competencia": wavg(d["pct_comp"], d["vol_map"]),
                    "% Demanda Perdida": wavg(d["pct_loss"], d["vol_map"]),
                }))
                .reset_index()
                .rename(columns={level_col: "Entidad"})
            )

            def clasificar(row):
                pi, pc, pl = row["% Interno"], row["% Competencia"], row["% Demanda Perdida"]

                # --- VALIDACIÓN BÁSICA ---
                if any(pd.isna([pi, pc, pl])):
                    return "⚪ Sin datos suficientes"

                # --- ESCENARIOS DE PÉRDIDA EXTERNA (riesgo competitivo) ---
                if pc >= 85:
                    return "🔴 Pérdida externa crítica (>85%)"
                elif 70 <= pc < 85:
                    return "🟥 Pérdida externa alta (70–85%)"
                elif 50 <= pc < 70:
                    return "🟧 Pérdida competitiva relevante (50–70%)"
                elif 35 <= pc < 50 and pi < 40:
                    return "🟨 Riesgo competitivo moderado (35–50%)"

                # --- ESCENARIOS DE DEMANDA PERDIDA (erosión estructural) ---
                if pl >= 65:
                    return "⚫ Demanda perdida crítica (>65%)"
                elif 45 <= pl < 65:
                    return "⚫ Demanda perdida alta (45–65%)"
                elif 25 <= pl < 45:
                    return "🟠 Demanda perdida moderada (25–45%)"
                elif 10 <= pl < 25 and pi < 50 and pc < 50:
                    return "🟤 Riesgo leve de erosión de demanda (10–25%)"

                # --- ESCENARIOS INTERNOS (retención o canibalización) ---
                if pi >= 95:
                    return "🟢 Canibalización interna total (>95%)"
                elif 85 <= pi < 95:
                    return "🟩 Canibalización interna fuerte (85–95%)"
                elif 70 <= pi < 85 and pc < 25:
                    return "🟢 Retención interna dominante (70–85%)"
                elif 50 <= pi < 70 and pc < 35:
                    return "🟢 Retención interna moderada (50–70%)"
                elif 30 <= pi < 50 and pc < 30 and pl < 30:
                    return "🟢 Retención interna débil (30–50%)"

                # --- ESCENARIOS MIXTOS (flujo combinado) ---
                if 40 <= pi < 80 and 20 <= pc < 60:
                    return "🟡 Flujo mixto interno–competencia (ambos >20%)"
                elif 30 <= pi < 60 and 15 <= pl < 40:
                    return "🟤 Flujo mixto interno–demanda perdida"
                elif 30 <= pc < 60 and 15 <= pl < 40:
                    return "🟣 Flujo mixto competencia–demanda perdida"

                # --- ESCENARIO ESTABLE ---
                if pi < 30 and pc < 30 and pl < 20:
                    return "⚪ Estable sin movimientos relevantes"

                return "⚪ Indeterminado"

            agg["Clasificación estratégica"] = agg.apply(clasificar, axis=1)
            agg = agg.round(2)

            return agg[[
                "Entidad", "Elasticidad Bruta", "Elasticidad Neta",
                "% Interno", "% Competencia", "% Demanda Perdida",
                "Clasificación estratégica"
            ]]

        except Exception as e:
            print("Error en sensibilidad_nivel_tabla_df:", e)
            return pd.DataFrame()


    # ----------------------------
    # Render de tabla
    # ----------------------------
    @render.data_frame
    def tabla_sensibilidad():
        """
        Tabla dinámica de Sensibilidad de Precio y Volumen.
        Adapta las columnas dinámicamente según el nivel detectado.
        """
        df = sensibilidad_nivel_tabla_df().copy()

        if df.empty or "Entidad" not in df.columns:
            print("⚠️ No se encontraron datos para los filtros actuales.")
            return pd.DataFrame(columns=[
                "Nombre", "Elasticidad Bruta", "Elasticidad Neta",
                "% Interno", "% Competencia", "% Demanda Perdida",
                "Clasificación estratégica"
            ])

        # Renombrar para visualización
        df = df.rename(columns={"Entidad": "Nombre"})

        # Forzar numéricos
        for col in ["Elasticidad Bruta", "Elasticidad Neta", "% Interno", "% Competencia", "% Demanda Perdida"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

        # Si hay menos columnas, completar
        for col in ["% Interno", "% Competencia", "% Demanda Perdida"]:
            if col not in df.columns:
                df[col] = np.nan

        # Reordenar columnas presentes
        cols = [
            "Nombre", "Elasticidad Bruta", "Elasticidad Neta",
            "% Interno", "% Competencia", "% Demanda Perdida",
            "Clasificación estratégica"
        ]
        df = df[[c for c in cols if c in df.columns]]

        # Si sigue vacío, mostrar aviso
        if df.empty:
            print("⚠️ sensibilidad_nivel_tabla_df devolvió datos vacíos tras filtrado.")
        return df
    
    @render_widget
    def fig_sankey():
        """
        Muestra un diagrama Sankey dinámico basado en el nivel seleccionado (cat/sub/marca).
        - Origen: entidad actual (ej. Agua con gas)
        - Destino: fabricante destino (COCA, PEPSICO, BONAFONT, etc.)
        - Valor: número de SKUs destino únicos con VOLUME_TRANSFER > 0
        """
        import plotly.graph_objects as go
        import re

        vm = df_vtm_map().copy()
        f = filters()

        # --- Filtrar solo Coca-Cola fabricante origen ---
        vm["SRC_MFG_KEY"] = _norm_series(vm.get("SRC_MFG", pd.Series(index=vm.index, dtype=str)))
        vm = vm[vm["SRC_MFG_KEY"].str.contains("COCA COLA FABRICANTE", na=False)]
        if vm.empty:
            return go.Figure()

        # --- Limpiar y filtrar por volumen positivo ---
        vm["VOLUME_TRANSFER"] = pd.to_numeric(vm["VOLUME_TRANSFER"], errors="coerce").fillna(0)
        vm = vm[vm["VOLUME_TRANSFER"] > 0].copy()
        if vm.empty:
            return go.Figure()

        # --- Limpieza robusta de textos ---
        def clean_text(s):
            if not isinstance(s, str):
                return ""
            s = s.upper().strip()
            s = re.sub(r"[^A-Z0-9ÁÉÍÓÚÑ ]+", " ", s)   # solo letras/números
            s = re.sub(r"\s+", " ", s)                 # quitar dobles espacios
            return s.strip()

        for col in ["CATEGORIA", "SUBCATEGORIA", "MARCA", "DEST_MFG", "DESTINATION_SKU"]:
            if col in vm.columns:
                vm[col] = vm[col].astype(str).apply(clean_text)

        # --- Aplicar filtros jerárquicos ---
        if f["cat"] != "All" and "CATEGORIA" in vm.columns:
            vm = vm[_norm_series(vm["CATEGORIA"]) == _norm_val(f["cat"])]
        if f["sub"] != "All" and "SUBCATEGORIA" in vm.columns:
            vm = vm[_norm_series(vm["SUBCATEGORIA"]) == _norm_val(f["sub"])]
        if f["marca"] != "All" and "MARCA" in vm.columns:
            vm = vm[_norm_series(vm["MARCA"]) == _norm_val(f["marca"])]

        if vm.empty:
            return go.Figure()

        # --- Normalización de columnas destino ---
        vm["DEST_MFG_CLEAN"] = vm["DEST_MFG"].fillna("DESCONOCIDO").apply(clean_text)
        vm["DESTINATION_SKU"] = vm["DESTINATION_SKU"].astype(str).apply(clean_text)

        # Agrupar por fabricante destino y SKU origen únicos
        df_links = (
            vm.groupby(["DEST_MFG_CLEAN", "SOURCE_SKU"], dropna=False)
            .size()
            .reset_index()
            .groupby("DEST_MFG_CLEAN")["SOURCE_SKU"]
            .nunique()
            .reset_index()
            .rename(columns={"SOURCE_SKU": "NUM_SKUS_ORIGEN"})
        )

        # --- Construcción del título jerárquico dinámico ---
        jerarquia = []
        if f["cat"] != "All": jerarquia.append(f["cat"])
        if f["sub"] != "All": jerarquia.append(f["sub"])
        if f["marca"] != "All": jerarquia.append(f["marca"])
        origen = " → ".join(jerarquia) if jerarquia else "COCA-COLA FABRICANTE"

        df_links["source"] = origen
        df_links["target"] = df_links["DEST_MFG_CLEAN"]
        df_links["value"] = df_links["NUM_SKUS_ORIGEN"]

        # --- Crear lista de nodos únicos ---
        labels = pd.unique(df_links[["source", "target"]].values.ravel("K")).tolist()
        label_to_id = {label: i for i, label in enumerate(labels)}
        df_links["source_id"] = df_links["source"].map(label_to_id)
        df_links["target_id"] = df_links["target"].map(label_to_id)

        # --- Ajuste de altura dinámico según cantidad de flujos ---
        altura_base = 450
        extra = len(df_links) * 25 
        altura_total = min(altura_base + extra, 1000)

        # --- Crear gráfico Sankey ---
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=25,
                thickness=20,
                line=dict(color="black", width=0.3),
                label=labels,
                color=[
                    "#E41E26" if "COCA" in lbl else "#4B4B4B"
                    for lbl in labels
                ]
            ),
            link=dict(
                source=df_links["source_id"],
                target=df_links["target_id"],
                value=df_links["value"],
                color=[
                    "rgba(228,30,38,0.4)" if "COCA" in str(t)
                    else "rgba(75,75,75,0.25)"
                    for t in df_links["target"]
                ],
                hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>"
                            "SKUs Destino: %{value}<extra></extra>"
            )
        )])

        # --- Layout estético ---
        fig.update_layout(
            title=dict(
                text=f"{origen}",
                x=0.5,
                xanchor="center",
                font=dict(size=16, family="Montserrat", color="#2E2E2E")
            ),
            height=altura_total,
            margin=dict(l=50, r=50, t=80, b=40),
            font=dict(size=13),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        return fig



    @render.download(filename="sensibilidad_precio_volumen.csv")
    def dl_sensibilidad():
        df = sensibilidad_nivel_tabla_df().copy()
        if df.empty:
            yield b""
        else:
            yield df.to_csv(index=False).encode("utf-8")

    # KPIs Riesgo Estratégico Agregado Sensibilidad Precio y volumen
    @render.ui
    def kpi_riesgo_externo():
        df = sensibilidad_nivel_tabla_df().copy()
        if df.empty or "% Competencia" not in df.columns:
            return kpi_card("% SKUs con pérdida externa (>50%)", "—")

        val = (df["% Competencia"] >= 50).mean() * 100
        color = "#E41E26" if val > 20 else "#2ECC71"
        return kpi_card(
            "% SKUs con pérdida externa (>50%)",
            ui.HTML(f"<span style='color:{color}'>{val:.1f}%</span>")
        )

    @render.ui
    def kpi_demanda_perdida():
        df = sensibilidad_nivel_tabla_df().copy()
        if df.empty or "% Demanda Perdida" not in df.columns:
            return kpi_card("% SKUs con demanda perdida (>30%)", "—")

        val = (df["% Demanda Perdida"] >= 30).mean() * 100
        color = "#E67E22" if val > 10 else "#2ECC71"
        return kpi_card(
            "% SKUs con demanda perdida (>30%)",
            ui.HTML(f"<span style='color:{color}'>{val:.1f}%</span>")
        )


    @render.ui
    def kpi_canibal_total():
        df = sensibilidad_nivel_tabla_df().copy()
        if df.empty or "% Interno" not in df.columns:
            return kpi_card("% SKUs con canibalización interna total (>90%)", "—")

        val = (df["% Interno"] >= 90).mean() * 100
        color = "#F1C40F" if val > 30 else "#2ECC71"
        return kpi_card(
            "% SKUs con canibalización interna total (>90%)",
            ui.HTML(f"<span style='color:{color}'>{val:.1f}%</span>")
        )

    @render.ui
    def kpi_indice_riesgo_global():
        df = sensibilidad_nivel_tabla_df().copy()
        if df.empty or not set(["% Competencia", "% Demanda Perdida", "% Interno"]).issubset(df.columns):
            return kpi_card("Índice de Riesgo Global (IRG)", "—")

        val = (
            0.5 * df["% Competencia"].mean() +
            0.3 * df["% Demanda Perdida"].mean() -
            0.2 * df["% Interno"].mean()
        )
        color = "#E41E26" if val > 0 else "#2ECC71"
        return kpi_card(
            "Índice de Riesgo Global (IRG)",
            ui.HTML(f"<span style='color:{color}'>{val:.2f}</span>")
        )
    
    # ================= Línea de tiempo (ajustada para incluir 'Marca')
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

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "Marca": "MARCA", "SKU": "SKU"}
        serie = monthly_eps_weighted(work, col_eps, entity_map.get(level_sel()))
        return kpi_card("Meses con elasticidad calculada", f"{int(serie['eps'].notna().sum()):,}")

    @render.ui
    def kpi_2():
        ctx = vtm_context()
        j = ctx["joined"]
        if j.empty: return kpi_card("PPG_PE ponderada (bruta)", "—")
        val = _wavg_metric_at_level(j, "PPG_PE", filters())
        return kpi_card("PPG_PE ponderada (bruta)", f"{val:.4f}" if pd.notna(val) else "—")

    @render.ui
    def kpi_3():
        ctx = vtm_context()
        j = ctx["joined"]
        if j.empty: return kpi_card("NET_PE ponderada (neta)", "—")
        val = _wavg_metric_at_level(j, "NET_PE", filters())
        return kpi_card("NET_PE ponderada (neta)", f"{val:.4f}" if pd.notna(val) else "—")
    
    @render.ui
    def kpi_4():
        """Elasticidad promedio ponderada según el modo (Teórica / Ajustada)"""
        base = df_joined().copy()
        f = filters()

        # Aplicar filtros activos
        if f['cat'] != "All" and "CATEGORIA" in base.columns: base = base[base["CATEGORIA"] == f['cat']]
        if f['sub'] != "All" and "SUBCATEGORIA" in base.columns: base = base[base["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA" in base.columns: base = base[base["MARCA"] == f['marca']]
        if f['sku'] != "All" and "SKU" in base.columns: base = base[base["SKU"] == f['sku']]

        win_start, win_end = last_full_year_window(df_base())
        base = base[(base["PERIOD"] >= win_start) & (base["PERIOD"] < win_end)].copy()
        if base.empty:
            return kpi_card("Elasticidad promedio ponderada", "—")

        # Calcular ε según modo
        usar_ajustada = (input.modo_eps() == "Teórica Ajustada")
        if usar_ajustada:
            work = apply_ranges_block(base.copy())
            col_eps = "epsilon_used"
        else:
            if "epsilon_teor" not in base.columns:
                dln_p = pd.to_numeric(base.get("dln_precio"), errors="coerce")
                dln_q = pd.to_numeric(base.get("dln_cu"), errors="coerce")
                base["epsilon_teor"] = (dln_q / dln_p).replace([np.inf, -np.inf], np.nan)
            work = base.copy()
            col_eps = "epsilon_teor"

        # Ponderación por volumen previo
        work["CU_lag"] = work.groupby("SKU", sort=False)["CU"].shift(1)
        w = pd.to_numeric(work["CU_lag"], errors="coerce").fillna(0)
        v = pd.to_numeric(work[col_eps], errors="coerce")

        mask = v.notna() & w.notna() & (w > 0)
        if not mask.any():
            return kpi_card("Elasticidad promedio ponderada", "—")

        val = float(np.average(v[mask], weights=w[mask]))
        estado = "Elástico" if abs(val) > 1 else "Inelástico"
        color = COCA_RED if abs(val) > 1 else COCA_GRAY_DARK

        etiqueta = f"{'Ajustada' if usar_ajustada else 'Teórica'} ({estado})"
        return kpi_card(f"Elasticidad promedio ponderada", ui.HTML(f"<span style='color:{color}'>{val:.3f}</span><br><span class='muted'>{etiqueta}</span>"))


    @render_widget
    def fig_linea():
        base = df_joined().copy()
        f = filters()

        # Aplicar filtros activos
        if f['cat'] != "All" and "CATEGORIA" in base.columns: base = base[base["CATEGORIA"] == f['cat']]
        if f['sub'] != "All" and "SUBCATEGORIA" in base.columns: base = base[base["SUBCATEGORIA"] == f['sub']]
        if f['marca'] != "All" and "MARCA" in base.columns: base = base[base["MARCA"] == f['marca']]
        if f['sku'] != "All" and "SKU" in base.columns: base = base[base["SKU"] == f['sku']]

        # Últimos 24 meses
        win_start, win_end = last_full_year_window(df_base())
        base = base[(base["PERIOD"] >= win_start) & (base["PERIOD"] < win_end)].copy()
        if base.empty: return go.Figure()

        for c in ["dln_precio", "dln_cu", "FINAL_PE", "CU", "P1_MIN", "P1_MAX", "P2_MIN", "P2_MAX"]:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")

        base["CU_lag"] = base.groupby("SKU", sort=False)["CU"].shift(1)
        base["PERIOD_M"] = pd.to_datetime(base["PERIOD"]).dt.to_period("M").dt.to_timestamp()

        # Selección de modo
        usar_ajustada = (input.modo_eps() == "Teórica Ajustada")
        if usar_ajustada:
            work = apply_ranges_block(base.copy())
            col_eps = "epsilon_used"
            linea_label = "Elasticidad Teórica Ajustada"
        else:
            base["epsilon_teor"] = (pd.to_numeric(base["dln_cu"], errors="coerce") /
                                    pd.to_numeric(base["dln_precio"], errors="coerce")).replace([np.inf, -np.inf], np.nan)
            work = base.copy()
            col_eps = "epsilon_teor"
            linea_label = "Elasticidad Teórica"

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "Marca": "MARCA", "SKU": "SKU"}
        serie = monthly_eps_weighted(work, col_eps, entity_map.get(level_sel()))

        ctx = vtm_context()
        j = ctx["joined"]
        w_ppg = _wavg_metric_at_level(j, "PPG_PE", filters())
        w_net = _wavg_metric_at_level(j, "NET_PE", filters())

        def wavg(df, col):
            v = pd.to_numeric(df[col], errors="coerce")
            w = pd.to_numeric(df.get("CU_lag", 0), errors="coerce").fillna(0)
            mask = v.notna() & w.notna() & (w > 0)
            return float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan

        p1_min, p1_max = wavg(base, "P1_MIN"), wavg(base, "P1_MAX")
        p2_min, p2_max = wavg(base, "P2_MIN"), wavg(base, "P2_MAX")

        # === GRÁFICO
        serie["PERIOD_M"] = pd.to_datetime(serie["PERIOD_M"], errors="coerce")
        x_dates = serie["PERIOD_M"].dt.tz_localize(None).dt.to_pydatetime()

        fig = go.Figure()

        # Línea principal
        fig.add_trace(go.Scatter(
            x=x_dates, y=serie["eps"],
            mode="lines+markers", name=linea_label,
            line=dict(width=2, color=COCA_RED),
            hovertemplate="%{x|%b %Y}<br>ε: %{y:.4f}<extra></extra>"
        ))

        # RANGOS SOMBREADOS
        work = apply_ranges_block(base.copy())

        # Detectar tipo de rango predominante (P1 o P2)
        if "PE_RANGE_CHECK_OVERALL" in work.columns:
            range_check = pd.to_numeric(work["PE_RANGE_CHECK_OVERALL"], errors="coerce")
            rango_predominante = 1 if (range_check == 1).sum() >= (range_check == 2).sum() else 2
        else:
            rango_predominante = 1

        # Seleccionar límites promedio ponderados del rango activo
        def _safe_avg(col):
            return pd.to_numeric(work[col], errors="coerce").mean(skipna=True)

        if rango_predominante == 1:
            rango_min, rango_max = _safe_avg("P1_MIN"), _safe_avg("P1_MAX")
            label, color = "P1 Range", "rgba(255, 0, 0, 0.08)"
        else:
            rango_min, rango_max = _safe_avg("P2_MIN"), _safe_avg("P2_MAX")
            label, color = "P2 Range", "rgba(0, 0, 0, 0.06)"

        # Dibujar solo el rango activo
        if not np.isnan(rango_min) and not np.isnan(rango_max):
            fig.add_shape(
                type="rect", xref="paper", yref="y",
                x0=0, x1=1, y0=rango_min, y1=rango_max,
                fillcolor=color, line=dict(width=0), layer="below"
            )
            fig.add_annotation(
                text=f"{label}: {rango_min:.2f} → {rango_max:.2f}",
                xref="paper", x=1.02, y=(rango_min + rango_max) / 2,
                yref="y", showarrow=False,
                font=dict(size=11, color="#4B4B4B")
            )

        # Líneas de referencia fuera del área
        for y, text, dash in [
            (w_ppg, "PPG_PE (bruta)", "dot"),
            (w_net, "NET_PE (neta)", "dash")
        ]:
            if pd.notna(y):
                fig.add_hline(
                    y=y,
                    line_width=2,
                    line_dash=dash,
                    annotation_text=text,
                    annotation_position="right",  # <-- corregido
                    annotation=dict(font=dict(size=11, color="#4B4B4B")),
        )

        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#AAAAAA")

        fig.update_layout(
            title=f"{linea_label} (último año) — Nivel: {level_sel()}",
            yaxis_title="Elasticidad",
            hovermode="x unified",
            height=CHART_H + 60,
            margin=dict(l=20, r=120, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
            plot_bgcolor="#FFFFFF",
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

        entity_map = {"Total": None, "Categoría": "CATEGORIA", "Subcategoría": "SUBCATEGORIA", "Marca": "MARCA", "SKU": "SKU"}
        serie = monthly_eps_weighted(work, col_eps, entity_map.get(level_sel()))

        ctx = vtm_context(); j = ctx["joined"]
        w_ppg = _wavg_metric_at_level(j, "PPG_PE", filters())
        w_net = _wavg_metric_at_level(j, "NET_PE", filters())

        out = serie.rename(columns={"PERIOD_M": "PERIOD", "eps": "eps_mensual"}).copy()
        out["PPG_PE_ref"] = w_ppg
        out["NET_PE_ref"] = w_net
        yield out.to_csv(index=False).encode("utf-8")

    # ================= Clasificación
    def classify_elasticity(e):
        if pd.isna(e): return "Sin dato"
        ae = abs(float(e))
        if math.isclose(ae, 1.0, rel_tol=1e-3, abs_tol=1e-3): return "Unitario"
        return "Inelástico (<1)" if ae < 1 else "Elástico (>1)"

    # --- Variable reactiva global para guardar el orden del eje Y ---
    order_y_reactive = reactive.Value([])


    @render_widget
    def fig_mapa():
        mt = df_mt_pe().copy()
        if mt.empty:
            return go.Figure()

        f = filters()

        # --- Aplicar filtros jerárquicos ---
        for col, key in [("CATEGORIA", "cat"), ("SUBCATEGORIA", "sub"), ("MARCA", "marca"), ("SKU", "sku")]:
            if f[key] != "All" and col in mt.columns:
                mt = mt[_norm_series(mt[col]) == _norm_val(f[key])]
        if mt.empty:
            return go.Figure()

        # --- Determinar nivel actual ---
        lvl = level_sel()
        if lvl == "Total":
            entity_col, nivel_txt = "CATEGORIA", "Categoría"
        elif lvl == "Categoría":
            entity_col, nivel_txt = "SUBCATEGORIA", "Subcategoría"
        elif lvl == "Subcategoría":
            entity_col, nivel_txt = "MARCA", "Marca"
        else:
            entity_col, nivel_txt = "SKU", "SKU"

        # --- Clasificación de elasticidad ---
        def classify_elasticity(e):
            if pd.isna(e):
                return "Sin dato"
            ae = abs(float(e))
            if math.isclose(ae, 1.0, rel_tol=1e-3, abs_tol=1e-3):
                return "Unitario"
            return "Inelástico (<1)" if ae < 1 else "Elástico (>1)"

        # --- Agregación dinámica ---
        if entity_col != "SKU":
            # niveles agregados: Categoría / Subcategoría / Marca
            agg_df = (
                mt.groupby(entity_col, dropna=False)
                .agg(
                    Elasticidad_prom=("NET_PE", "mean"),
                    Elasticidad_bruta=("PPG_PE", "mean"),
                    SKUs_total=("SKU", "nunique"),
                    SKUs_elasticos=("NET_PE", lambda s: (abs(s) > 1).sum()),
                    SKUs_inelasticos=("NET_PE", lambda s: (abs(s) < 1).sum())
                )
                .reset_index()
            )
        else:
            # nivel SKU: solo métricas individuales
            agg_df = (
                mt.groupby(entity_col, dropna=False)
                .agg(
                    Elasticidad_prom=("NET_PE", "mean"),
                    Elasticidad_bruta=("PPG_PE", "mean"),
                )
                .reset_index()
            )

        agg_df["Clase"] = agg_df["Elasticidad_prom"].apply(classify_elasticity)

        # --- Orden visual unificado ---
        order_y = agg_df.sort_values("Elasticidad_prom", ascending=False)[entity_col].tolist()
        order_y_reactive.set(order_y)

        color_map = {
            "Elástico (>1)": COCA_RED,
            "Unitario": "#000000",
            "Inelástico (<1)": COCA_GRAY_DARK,
            "Sin dato": "#B3B3B3",
        }

        # --- Crear figura ---
        fig = go.Figure()

        for cls in ["Elástico (>1)", "Unitario", "Inelástico (<1)"]:
            sub = agg_df[agg_df["Clase"] == cls]
            if sub.empty:
                continue

            # Tooltip condicional según el nivel
            if entity_col != "SKU":
                hovertemplate = (
                    "<b>%{y}</b><br>"
                    "Elasticidad neta: %{x:.3f}<br>"
                    "Elasticidad bruta: %{customdata[0]:.3f}<br>"
                    "SKUs elásticos: %{customdata[1]}<br>"
                    "SKUs inelásticos: %{customdata[2]}<br>"
                    "Total de SKUs: %{customdata[3]}<extra></extra>"
                )
                customdata = np.stack([
                    sub["Elasticidad_bruta"],
                    sub.get("SKUs_elasticos", np.nan),
                    sub.get("SKUs_inelasticos", np.nan),
                    sub.get("SKUs_total", np.nan)
                ], axis=-1)
            else:
                hovertemplate = (
                    "<b>%{y}</b><br>"
                    "Elasticidad neta (NET_PE): %{x:.3f}<br>"
                    "Elasticidad bruta (PPG_PE): %{customdata[0]:.3f}<extra></extra>"
                )
                customdata = np.stack([
                    sub["Elasticidad_bruta"]
                ], axis=-1)

            fig.add_trace(go.Bar(
                orientation="h",
                x=sub["Elasticidad_prom"],
                y=sub[entity_col].astype(str),
                name=cls,
                marker_color=color_map.get(cls, "#B3B3B3"),
                hovertemplate=hovertemplate,
                customdata=customdata
            ))

        # --- Layout elegante ---
        fig.update_layout(
            title=dict(
                text=f"<b>Elasticidad Promedio Neta por {nivel_txt}</b>",
                x=0.47, xanchor="center",
                font=dict(size=18, family="Arial", color="#2E2E2E")
            ),
            xaxis_title="Elasticidad (ε)",
            yaxis_title=nivel_txt,
            yaxis=dict(categoryorder="array", categoryarray=order_y, automargin=True),
            height=450,
            showlegend=False,
            margin=dict(l=90, r=60, t=70, b=50),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF"
        )

        # Línea de referencia (-1)
        fig.add_vline(x=-1, line_width=2, line_dash="dot", line_color="#999999")

        # Ajuste dinámico del rango X
        min_x = min(agg_df["Elasticidad_prom"].min() - 0.2, -2)
        fig.update_xaxes(range=[min_x, 0])

        return fig



    @render_widget
    def fig_pie():
        mt = df_mt_pe().copy()
        if mt.empty:
            return go.Figure()

        f = filters()
        for col, key in [("CATEGORIA", "cat"), ("SUBCATEGORIA", "sub"), ("MARCA", "marca"), ("SKU", "sku")]:
            if f[key] != "All" and col in mt.columns:
                mt = mt[_norm_series(mt[col]) == _norm_val(f[key])]
        if mt.empty or "NET_PE" not in mt.columns:
            return go.Figure()

        def clasificar(e):
            e = pd.to_numeric(e, errors="coerce")
            if pd.isna(e):
                return "Sin dato"
            if abs(e - 1) < 1e-3:
                return "Unitario"
            return "Elástico (>1)" if abs(e) > 1 else "Inelástico (<1)"

        mt["Clase"] = mt["NET_PE"].apply(clasificar)

        lvl = level_sel()
        entity_col = (
            "CATEGORIA" if lvl == "Total" else
            "SUBCATEGORIA" if lvl == "Categoría" else
            "MARCA" if lvl == "Subcategoría" else
            "SKU"
        )

        dist = (
            mt.groupby(entity_col)["Clase"]
            .value_counts(normalize=True)
            .rename("Porcentaje")
            .mul(100)
            .reset_index()
        )

        color_map = {
            "Elástico (>1)": COCA_RED,
            "Inelástico (<1)": COCA_GRAY_DARK,
            "Unitario": "#000000",
            "Sin dato": "#B3B3B3",
        }

        fig = px.bar(
            dist,
            x="Porcentaje",
            y=entity_col,
            color="Clase",
            orientation="h",
            color_discrete_map=color_map,
            text=dist["Porcentaje"].map(lambda v: f"{v:.1f}%")
        )

        fig.update_traces(
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate="<b>%{y}</b><br>%{color}: %{x:.1f}%<extra></extra>"
        )

        order_y = order_y_reactive.get() or sorted(dist[entity_col].unique().tolist())

        fig.update_layout(
            title=dict(
                text="<b>Distribución Elásticos e Inelásticos</b>",
                x=0.47, xanchor="center",
                font=dict(size=18, family="Arial", color="#2E2E2E")
            ),
            barmode="stack",
            xaxis_title="Porcentaje de SKUs",
            yaxis_title=None,
            yaxis=dict(categoryorder="array", categoryarray=order_y),
            height=450,
            showlegend=False,
            margin=dict(l=40, r=60, t=70, b=50),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF"
        )
        return fig

    @render.download(filename="mapa_elasticidad.csv")
    def dl_mapa():
        dj = df_joined().copy()
        if dj.empty: yield b""; return
        yield dj.to_csv(index=False).encode("utf-8")

    # ================= Explorador (agrega 'Marca')
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
                              xaxis=dict(showgrid=False), height=CHART_H, margin=dict(l=10, r=10, t=30, b=10))
            fig.add_hline(y=0, line_width=1, line_dash="dot")
            return fig
        else:
            group = ["PERIOD"]; entity_col = None
            lvl = level_sel()
            if lvl=="Categoría" and "CATEGORIA" in d.columns:
                group.append("CATEGORIA"); entity_col = "CATEGORIA"
            elif lvl=="Subcategoría" and "SUBCATEGORIA" in d.columns:
                group.append("SUBCATEGORIA"); entity_col = "SUBCATEGORIA"
            elif lvl=="Marca" and "MARCA" in d.columns:
                group.append("MARCA"); entity_col = "MARCA"
            elif lvl=="SKU" and "SKU" in d.columns:
                group.append("SKU"); entity_col = "SKU"
            agg = d.groupby(group, dropna=False)[metric_col].sum().reset_index()
            titulo_metric = "CU" if metric_col == "CU" else "Ingreso Neto"
            fig = px.line(agg, x="PERIOD", y=metric_col, color=(entity_col if entity_col in (agg.columns) else None),
                          title=f"{lvl} mensual — {titulo_metric}")
            fig.update_traces(mode="lines", line=dict(width=2), hovertemplate="<b>%{fullData.name}</b><br>%{x|%Y-%m}: %{y:,}")
            fig.update_layout(title=dict(x=0.01), hovermode="x unified", showlegend=False, xaxis=dict(showgrid=False),
                              yaxis=dict(showgrid=True, gridcolor=COCA_GRAY_LIGHT), height=CHART_H, margin=dict(l=10, r=10, t=30, b=10))
            return fig

    @render.download(filename=lambda: f"{level_sel()}_{'cu' if input.metric_mode()=='Volumen (CU)' else 'ingreso_neto'}.csv")
    def dl_explorer():
        d = df_filtered().copy()
        yield d.to_csv(index=False).encode("utf-8")

    # ================= Bennett
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

# --- Inicialización con assets estáticos ---
app = App(
    app_ui,
    server,
    static_assets=os.path.join(os.path.dirname(__file__), "www")
)
