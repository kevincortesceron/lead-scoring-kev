"""
features.py — Limpieza, feature engineering y reglas de priorización.

Lo usan el notebook (entrenamiento), la API (FastAPI) y el dashboard (Streamlit),
para que el modelo reciba exactamente las mismas transformaciones en todos lados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "Converted"
ID_COLS = ["Prospect ID", "Lead Number"]

# Columnas que se llenan DESPUÉS del contacto comercial (fuga de información / data leakage).
# Ej.: Tags = "Closed by Horizzon" convierte 99%; "Lead Quality" la asigna el vendedor.
# Un modelo de lead scoring debe puntuar al lead cuando llega, así que se excluyen.
LEAKY_COLS = [
    "Tags",
    "Lead Quality",
    "Lead Profile",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
    "Last Notable Activity",  # redundante con Last Activity y se actualiza en el tiempo
]

# Columnas casi constantes (>99% un solo valor) o con demasiados faltantes.
LOW_INFO_COLS = [
    "Do Not Call", "Search", "Magazine", "Newspaper Article", "X Education Forums",
    "Newspaper", "Digital Advertisement", "Through Recommendations",
    "Receive More Updates About Our Courses", "Update me on Supply Chain Content",
    "Get updates on DM Content", "I agree to pay the amount through cheque",
    "What matters most to you in choosing a course",  # 99.9% "Better Career Prospects"
    "How did you hear about X Education",  # 78% faltante
]

# Campos crudos que el modelo necesita como entrada (lo que llega de un formulario / CRM).
INPUT_COLS = [
    "Lead Origin",
    "Lead Source",
    "Do Not Email",
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Last Activity",
    "Country",
    "Specialization",
    "What is your current occupation",
    "City",
    "A free copy of Mastering The Interview",
]

NUMERIC_FEATURES = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "log_time_on_site",
    "time_per_visit",
    "total_page_views",
    "engagement_index",
    "do_not_email",
    "free_copy",
    "is_india",
    "occupation_missing",
    "specialization_missing",
    "city_missing",
    "is_working_professional",
    "high_intent_activity",
    "no_activity_visit",
]

CATEGORICAL_FEATURES = [
    "Lead Origin",
    "Lead Source",
    "Last Activity",
    "Specialization",
    "What is your current occupation",
    "City",
]

# Topes para outliers (≈ percentil 99 del dataset de entrenamiento).
CAPS = {"TotalVisits": 17, "Page Views Per Visit": 9, "Total Time Spent on Website": 2272}

# Actividades que indican intención alta de compra.
HIGH_INTENT = {"SMS Sent", "Had a Phone Conversation", "Email Link Clicked", "Form Submitted on Website"}

SOURCE_FIXES = {"google": "Google", "Facebook": "Social Media", "bing": "Search Engines",
                "Click2call": "Other", "Press_Release": "Other", "Social Media": "Social Media",
                "Live Chat": "Olark Chat", "youtubechannel": "Social Media",
                "testone": "Other", "Pay per Click Ads": "Other", "welearnblog_Home": "Other",
                "WeLearn": "Other", "blog": "Other", "NC_EDM": "Other"}


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: 'Select' → nulo, nombres de fuentes unificados, tipos numéricos."""
    df = df.copy()
    df = df.replace("Select", np.nan)
    for c in INPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if "Lead Source" in df:
        df["Lead Source"] = df["Lead Source"].replace(SOURCE_FIXES)
    for c in ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


class LeadFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transforma registros crudos de leads en las features del modelo."""

    def fit(self, X, y=None):
        X = clean_raw(pd.DataFrame(X))
        self.medians_ = {
            c: float(X[c].median())
            for c in ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]
        }
        return self

    def transform(self, X):
        X = clean_raw(pd.DataFrame(X))
        out = pd.DataFrame(index=X.index)

        # Numéricas con imputación por mediana y tope de outliers
        for c, cap in CAPS.items():
            out[c] = X[c].fillna(self.medians_[c]).clip(lower=0, upper=cap)

        visits = out["TotalVisits"]
        time = out["Total Time Spent on Website"]
        ppv = out["Page Views Per Visit"]

        # Features derivadas de engagement
        out["log_time_on_site"] = np.log1p(time)
        out["time_per_visit"] = time / (visits + 1)
        out["total_page_views"] = visits * ppv
        out["engagement_index"] = (
            time / CAPS["Total Time Spent on Website"]
            + visits / CAPS["TotalVisits"]
            + ppv / CAPS["Page Views Per Visit"]
        ) / 3
        out["no_activity_visit"] = (visits == 0).astype(int)

        # Binarias
        yes = lambda s: s.astype(str).str.strip().str.lower().eq("yes").astype(int)
        out["do_not_email"] = yes(X["Do Not Email"])
        out["free_copy"] = yes(X["A free copy of Mastering The Interview"])
        out["is_india"] = X["Country"].fillna("India").eq("India").astype(int)

        # Indicadores de faltantes (el faltante también es información)
        out["occupation_missing"] = X["What is your current occupation"].isna().astype(int)
        out["specialization_missing"] = X["Specialization"].isna().astype(int)
        out["city_missing"] = X["City"].isna().astype(int)

        out["is_working_professional"] = X["What is your current occupation"].eq("Working Professional").astype(int)
        out["high_intent_activity"] = X["Last Activity"].isin(HIGH_INTENT).astype(int)

        # Categóricas (faltante → "Desconocido")
        for c in CATEGORICAL_FEATURES:
            out[c] = X[c].astype("object").where(X[c].notna(), "Desconocido").astype(str)

        return out[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# ---------------------------------------------------------------- Priorización
TIERS = [  # (score mínimo, nivel, acción sugerida)
    (70, "Caliente", "Llamar hoy: asignar a un asesor senior"),
    (40, "Tibio", "Contactar esta semana: llamada + email personalizado"),
    (0, "Frío", "Nurturing automático: secuencia de emails / SMS"),
]


def priority_from_score(score: float) -> tuple[str, str]:
    for threshold, tier, action in TIERS:
        if score >= threshold:
            return tier, action
    return TIERS[-1][1], TIERS[-1][2]


def score_leads(model, df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el DataFrame con score (0-100), prioridad, acción y posición en el ranking."""
    proba = np.asarray(model.predict_proba(df)[:, 1], dtype=float)
    out = df.copy()
    out["probabilidad"] = proba.round(4)
    out["score"] = (proba * 100).round(1)
    pr = out["score"].apply(priority_from_score)
    out["prioridad"] = [p[0] for p in pr]
    out["accion_sugerida"] = [p[1] for p in pr]
    out = out.sort_values("score", ascending=False)
    out["ranking"] = np.arange(1, len(out) + 1)
    return out
