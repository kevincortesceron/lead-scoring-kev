"""
API de Lead Scoring — FastAPI

Endpoints:
  GET  /health          estado del servicio
  GET  /model-info      métricas y metadatos del modelo
  POST /predict         puntúa un lead
  POST /predict/batch   puntúa varios leads y los devuelve ordenados (ranking)

Ejecutar local:  uvicorn api.main:app --reload   (desde la raíz del proyecto)
Documentación interactiva: /docs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import INPUT_COLS, score_leads  # noqa: E402

MODEL = joblib.load(ROOT / "models" / "lead_scoring_model.joblib")
METADATA = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))

app = FastAPI(
    title="Lead Scoring Predictivo API",
    description="Clasifica leads según su probabilidad de conversión y les asigna una prioridad comercial.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

YesNo = Literal["Yes", "No"]


class Lead(BaseModel):
    lead_id: Optional[str] = Field(None, description="Identificador del lead en tu CRM (opcional)")
    lead_origin: Optional[str] = Field(None, examples=["Landing Page Submission"])
    lead_source: Optional[str] = Field(None, examples=["Google"])
    do_not_email: YesNo = "No"
    total_visits: Optional[float] = Field(None, ge=0, examples=[5])
    total_time_spent_on_website: Optional[float] = Field(None, ge=0, description="Segundos", examples=[1200])
    page_views_per_visit: Optional[float] = Field(None, ge=0, examples=[3])
    last_activity: Optional[str] = Field(None, examples=["SMS Sent"])
    country: Optional[str] = Field(None, examples=["India"])
    specialization: Optional[str] = Field(None, examples=["Finance Management"])
    current_occupation: Optional[str] = Field(None, examples=["Working Professional"])
    city: Optional[str] = Field(None, examples=["Mumbai"])
    free_copy_mastering_interview: YesNo = "No"


API_TO_RAW = dict(zip(
    ["lead_origin", "lead_source", "do_not_email", "total_visits", "total_time_spent_on_website",
     "page_views_per_visit", "last_activity", "country", "specialization", "current_occupation",
     "city", "free_copy_mastering_interview"],
    INPUT_COLS,
))


class Prediction(BaseModel):
    lead_id: Optional[str]
    probabilidad: float
    score: float
    prioridad: str
    accion_sugerida: str
    ranking: int


def _predict(leads: list[Lead]) -> list[Prediction]:
    rows = [{API_TO_RAW[k]: v for k, v in lead.model_dump().items() if k in API_TO_RAW} for lead in leads]
    df = pd.DataFrame(rows, columns=INPUT_COLS)
    df["lead_id"] = [lead.lead_id for lead in leads]
    scored = score_leads(MODEL, df)
    return [
        Prediction(lead_id=r.lead_id, probabilidad=round(float(r.probabilidad), 4), score=round(float(r.score), 1),
                   prioridad=r.prioridad, accion_sugerida=r.accion_sugerida, ranking=int(r.ranking))
        for r in scored.itertuples()
    ]


@app.get("/")
def root():
    return {"servicio": "Lead Scoring Predictivo", "docs": "/docs", "modelo": METADATA["model_name"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {k: METADATA[k] for k in ["model_name", "trained_at", "oof_roc_auc", "metrics", "tiers",
                                      "feature_importance", "versions"]}


@app.post("/predict", response_model=Prediction)
def predict(lead: Lead):
    return _predict([lead])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(leads: list[Lead]):
    if not leads:
        raise HTTPException(400, "Envía al menos un lead")
    if len(leads) > 5000:
        raise HTTPException(413, "Máximo 5000 leads por petición")
    return _predict(leads)
