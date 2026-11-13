"""
Dashboard comercial — Lead Scoring Predictivo (Streamlit)

Ejecutar local:  streamlit run dashboard/app.py   (desde la raíz del proyecto)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import INPUT_COLS, TIERS, score_leads  # noqa: E402

st.set_page_config(page_title="Lead Scoring — Dashboard Comercial", page_icon="🎯", layout="wide")

TIER_ORDER = ["Caliente", "Tibio", "Frío"]
TIER_ICON = {"Caliente": "🔥", "Tibio": "🌤", "Frío": "❄️"}
# Rampa ordinal de un solo tono (más oscuro = más prioridad)
TIER_COLORS = {"Caliente": "#1c5cab", "Tibio": "#3987e5", "Frío": "#86b6ef"}
GRID = "rgba(128,128,128,0.15)"


@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models" / "lead_scoring_model.joblib")
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    return model, meta


@st.cache_data
def load_scored():
    return pd.read_csv(ROOT / "data" / "leads_scored.csv")


def style_fig(fig, height=360):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=90, b=10), title_y=0.97, title_yanchor="top",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", bargap=0.25,
                      legend=dict(orientation="h", y=1.03, yanchor="bottom", x=0, title=None),
                      hoverlabel=dict(bgcolor="white"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


model, meta = load_model()

# ------------------------------------------------------------------ Sidebar / datos
st.sidebar.title("🎯 Lead Scoring")
st.sidebar.caption(f"Modelo: **{meta['model_name']}** · ROC-AUC fuera de muestra **{meta['oof_roc_auc']:.3f}**")
source = st.sidebar.radio("Fuente de datos", ["Base histórica puntuada", "Subir CSV de leads nuevos"])

if source == "Subir CSV de leads nuevos":
    st.sidebar.caption("CSV con las columnas del dataset original (Lead Origin, Lead Source, TotalVisits, ...).")
    up = st.sidebar.file_uploader("Archivo CSV", type="csv")
    if up is None:
        st.info("Sube un CSV para puntuar leads nuevos, o elige la base histórica en la barra lateral.")
        st.stop()
    raw = pd.read_csv(up)
    missing = [c for c in INPUT_COLS if c not in raw.columns]
    if len(missing) == len(INPUT_COLS):
        st.error("El CSV no tiene ninguna de las columnas esperadas.")
        st.stop()
    if missing:
        st.warning(f"Columnas faltantes (se tratarán como desconocidas): {', '.join(missing)}")
    data = score_leads(model, raw)
else:
    data = load_scored()

has_target = "Converted" in data.columns
if "Lead Number" not in data.columns:
    data["Lead Number"] = np.arange(1, len(data) + 1)

st.sidebar.divider()
st.sidebar.subheader("Filtros")
f_tier = st.sidebar.multiselect("Prioridad", TIER_ORDER, default=TIER_ORDER)
sources = sorted(data["Lead Source"].dropna().astype(str).unique()) if "Lead Source" in data else []
f_source = st.sidebar.multiselect("Fuente del lead", sources, placeholder="Todas")
f_min = st.sidebar.slider("Score mínimo", 0, 100, 0)

view = data[data["prioridad"].isin(f_tier) & (data["score"] >= f_min)]
if f_source:
    view = view[view["Lead Source"].astype(str).isin(f_source)]

# ------------------------------------------------------------------ Encabezado y KPIs
st.title("Dashboard comercial de leads")
st.caption("Ranking y priorización de leads según su probabilidad de conversión.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Leads en vista", f"{len(view):,}")
for col, tier in zip([k2, k3, k4], TIER_ORDER):
    n = int((view["prioridad"] == tier).sum())
    col.metric(f"{TIER_ICON[tier]} {tier}", f"{n:,}", f"{n / max(len(view), 1):.0%} del total", delta_color="off", delta_arrow="off")
k5.metric("Conversiones esperadas", f"{view['probabilidad'].sum():,.0f}",
          f"{view['probabilidad'].mean():.0%} prob. media" if len(view) else None, delta_color="off", delta_arrow="off")

tab_rank, tab_an, tab_new, tab_model = st.tabs(["📋 Ranking", "📊 Análisis", "➕ Puntuar un lead", "🧠 Modelo"])

# ------------------------------------------------------------------ Ranking
with tab_rank:
    st.subheader("Ranking de leads priorizados")
    top_n = st.select_slider("Mostrar", options=[25, 50, 100, 250, 500, 1000], value=50)
    cols = ["ranking", "Lead Number", "score", "prioridad", "accion_sugerida", "Lead Origin", "Lead Source",
            "Last Activity", "What is your current occupation", "Total Time Spent on Website", "TotalVisits"]
    if has_target:
        cols.append("Converted")
    table = view.sort_values("score", ascending=False)[[c for c in cols if c in view.columns]].head(top_n).copy()
    table["prioridad"] = table["prioridad"].map(lambda t: f"{TIER_ICON[t]} {t}")
    st.dataframe(
        table, hide_index=True, width="stretch", height=480,
        column_config={
            "ranking": st.column_config.NumberColumn("#", width="small"),
            "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
            "accion_sugerida": st.column_config.TextColumn("Acción sugerida", width="large"),
            "Total Time Spent on Website": st.column_config.NumberColumn("Tiempo en sitio (s)"),
            "Converted": st.column_config.CheckboxColumn("Convirtió (histórico)"),
        },
    )
    st.download_button("⬇️ Descargar ranking filtrado (CSV)",
                       view.sort_values("score", ascending=False).to_csv(index=False).encode("utf-8"),
                       "ranking_leads.csv", "text/csv")

# ------------------------------------------------------------------ Análisis
with tab_an:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(view, x="score", color="prioridad", nbins=40, category_orders={"prioridad": TIER_ORDER},
                           color_discrete_map=TIER_COLORS, labels={"score": "Score", "count": "Leads"},
                           title="Distribución de scores")
        fig.update_traces(marker_line_width=1, marker_line_color="white",
                          hovertemplate="Score %{x}<br>%{y} leads<extra>%{fullData.name}</extra>")
        fig.update_yaxes(title="Leads")
        st.plotly_chart(style_fig(fig), width="stretch")
    with c2:
        if has_target and len(view):
            t = view.groupby("prioridad")["Converted"].agg(["mean", "size"]).reindex(TIER_ORDER).dropna().reset_index()
            fig = go.Figure(go.Bar(
                x=t["prioridad"], y=t["mean"], marker_color=[TIER_COLORS[p] for p in t["prioridad"]],
                text=[f"{m:.0%}" for m in t["mean"]], textposition="outside",
                customdata=t["size"], hovertemplate="%{x}<br>Conversión real: %{y:.1%}<br>%{customdata} leads<extra></extra>"))
            fig.add_hline(y=view["Converted"].mean(), line_dash="dash", line_color="gray",
                          annotation_text="Promedio general", annotation_position="top right")
            fig.update_layout(title="Conversión real por nivel de prioridad", showlegend=False)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.1], title="Tasa de conversión")
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            st.info("La conversión real por nivel aparece cuando los datos traen la columna `Converted`.")

    c3, c4 = st.columns(2)
    with c3:
        if "Lead Source" in view and len(view):
            s = (view.groupby("Lead Source").agg(leads=("score", "size"), score=("score", "mean"))
                 .query("leads >= 20").sort_values("score").tail(10).reset_index())
            fig = go.Figure(go.Bar(x=s["score"], y=s["Lead Source"], orientation="h", marker_color="#2a78d6",
                                   customdata=s["leads"],
                                   hovertemplate="%{y}<br>Score medio: %{x:.1f}<br>%{customdata} leads<extra></extra>"))
            fig.update_layout(title="Score medio por fuente (fuentes con ≥ 20 leads)")
            fig.update_xaxes(title="Score medio", range=[0, 100], gridcolor=GRID, showgrid=True)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(style_fig(fig), width="stretch")
    with c4:
        if has_target and view["Converted"].sum() > 0:
            g = view.sort_values("score", ascending=False)
            pct_leads = np.arange(1, len(g) + 1) / len(g) * 100
            pct_conv = g["Converted"].cumsum().values / g["Converted"].sum() * 100
            idx = np.linspace(0, len(g) - 1, min(len(g), 200)).astype(int)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.r_[0, pct_leads[idx]], y=np.r_[0, pct_conv[idx]], name="Con el modelo",
                                     line=dict(color="#2a78d6", width=2),
                                     hovertemplate="Contactando el %{x:.0f}% de leads<br>se captura el %{y:.0f}% de conversiones<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Sin modelo (azar)",
                                     line=dict(color="gray", width=2, dash="dash"), hoverinfo="skip"))
            fig.update_layout(title="Curva de ganancias: ¿a quién llamar primero?", hovermode="x unified")
            fig.update_xaxes(title="% de leads contactados (por score)", showgrid=True, gridcolor=GRID)
            fig.update_yaxes(title="% de conversiones capturadas")
            st.plotly_chart(style_fig(fig), width="stretch")
            k = int(len(g) * 0.3)
            st.caption(f"Llamando solo al 30% superior del ranking se captura el "
                       f"**{g['Converted'].iloc[:k].sum() / g['Converted'].sum():.0%}** de las conversiones.")

# ------------------------------------------------------------------ Puntuar un lead
with tab_new:
    st.subheader("Puntuar un lead nuevo")
    ref = load_scored()
    opts = lambda c: ["(desconocido)"] + sorted(ref[c].dropna().astype(str).unique())
    with st.form("nuevo_lead"):
        a, b, c = st.columns(3)
        origin = a.selectbox("Origen del lead", opts("Lead Origin"))
        src = a.selectbox("Fuente", opts("Lead Source"))
        last = a.selectbox("Última actividad", opts("Last Activity"))
        occ = b.selectbox("Ocupación", opts("What is your current occupation"))
        spec = b.selectbox("Especialización", opts("Specialization"))
        city = b.selectbox("Ciudad", opts("City"))
        visits = c.number_input("Visitas al sitio", 0, 300, 3)
        tsec = c.number_input("Tiempo en el sitio (segundos)", 0, 5000, 300, step=30)
        ppv = c.number_input("Páginas por visita", 0.0, 60.0, 2.0, step=0.5)
        d1, d2, d3 = st.columns(3)
        country = d1.text_input("País", "India")
        dne = d2.selectbox("¿No enviar emails?", ["No", "Yes"])
        free = d3.selectbox("¿Pidió copia gratis de 'Mastering The Interview'?", ["No", "Yes"])
        submitted = st.form_submit_button("Calcular score", type="primary")

    if submitted:
        nz = lambda v: None if v == "(desconocido)" else v
        row = pd.DataFrame([dict(zip(INPUT_COLS, [nz(origin), nz(src), dne, visits, tsec, ppv, nz(last), country,
                                                   nz(spec), nz(occ), nz(city), free]))])
        r = score_leads(model, row).iloc[0]
        g1, g2 = st.columns([1, 1.3])
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=r["score"], number={"suffix": " / 100"},
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": TIER_COLORS[r["prioridad"]], "thickness": 0.3},
                       "steps": [{"range": [0, 40], "color": "rgba(134,182,239,0.15)"},
                                 {"range": [40, 70], "color": "rgba(57,135,229,0.15)"},
                                 {"range": [70, 100], "color": "rgba(28,92,171,0.15)"}]}))
            st.plotly_chart(style_fig(fig, 260), width="stretch")
        with g2:
            st.markdown(f"### {TIER_ICON[r['prioridad']]} Prioridad: **{r['prioridad']}**")
            st.markdown(f"Probabilidad de conversión: **{r['probabilidad']:.1%}**")
            st.markdown(f"**Acción sugerida:** {r['accion_sugerida']}")
            pct = (ref["score"] < r["score"]).mean()
            st.caption(f"Este lead supera al {pct:.0%} de los leads históricos.")

# ------------------------------------------------------------------ Modelo
with tab_model:
    st.subheader("Comparación de modelos")
    st.dataframe(pd.DataFrame(meta["metrics"]), hide_index=True, width="stretch")
    st.caption("Métrica principal: ROC-AUC (validación cruzada de 5 folds + conjunto de prueba del 20%).")
    imp = pd.Series(meta["feature_importance"]).sort_values().tail(15)
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2a78d6",
                           hovertemplate="%{y}<br>Importancia: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Variables más importantes (XGBoost)")
    fig.update_xaxes(title="Importancia (gain)", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(style_fig(fig, 480), width="stretch")
    st.markdown("**Reglas de priorización**")
    st.table(pd.DataFrame([{"Nivel": f"{TIER_ICON[n]} {n}", "Score mínimo": t, "Acción": a} for t, n, a in TIERS]))
