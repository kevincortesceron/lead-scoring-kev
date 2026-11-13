# Lead Scoring Predictivo — X Education

Modelo que clasifica leads según su probabilidad de conversión, desplegado como **API (endpoint)** y **dashboard comercial**
con ranking y sistema de priorización.

| | |
|---|---|
| Mejor modelo | XGBoost (hiperparámetros con RandomizedSearchCV) |
| ROC-AUC | 0.907 CV · 0.890 test · 0.905 fuera de muestra (todo el dataset) |
| Modelos comparados | Regresión logística, Gradient Boosting, XGBoost, Red neuronal (MLP) |
| Priorización | 🔥 Caliente (≥70) convierte 86% · 🌤 Tibio (40–69) 58% · ❄️ Frío (<40) 12% |

## Estructura

```
├── Lead_Scoring_Predictivo.ipynb   # Notebook de Colab: EDA, features, modelos, ROC-AUC, ranking, exportación
├── features.py                     # Limpieza + feature engineering + reglas de priorización (compartido)
├── api/main.py                     # Endpoint FastAPI
├── dashboard/app.py                # Dashboard comercial Streamlit
├── models/                         # Modelo entrenado (.joblib) + metadata.json (métricas)
├── data/                           # Leads.csv (original) + leads_scored.csv (ranking)
├── requirements.txt · render.yaml · Dockerfile
```

## 1. Notebook en Google Colab

1. Sube el proyecto a un repositorio de GitHub.
2. En Colab: **Archivo → Abrir cuaderno → GitHub** y pega la URL del repo, o sube el `.ipynb` directamente.
3. **Entorno de ejecución → Ejecutar todo** (~2 min). El notebook descarga el dataset solo si no lo encuentra
   y regenera `models/` y `data/leads_scored.csv`.

## 2. Ejecutar en local

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload            # API en http://localhost:8000/docs
streamlit run dashboard/app.py           # Dashboard en http://localhost:8501
```

## 3. Desplegar el endpoint (Render, gratis)

1. Entra a [render.com](https://render.com) con tu cuenta de GitHub → **New → Blueprint** → elige el repo.
   Render lee `render.yaml` y crea el servicio `lead-scoring-api`.
2. Espera el build (~3–5 min). Tu endpoint queda en `https://lead-scoring-api-xxxx.onrender.com`.
3. Pruébalo en `/docs` (Swagger) o con:

```bash
curl -X POST https://TU-URL.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"lead_id":"CRM-001","lead_origin":"Landing Page Submission","lead_source":"Google",
       "total_visits":5,"total_time_spent_on_website":1200,"page_views_per_visit":3,
       "last_activity":"SMS Sent","current_occupation":"Working Professional","city":"Mumbai"}'
```

Respuesta:
```json
{"lead_id":"CRM-001","probabilidad":0.9642,"score":96.4,"prioridad":"Caliente",
 "accion_sugerida":"Llamar hoy: asignar a un asesor senior","ranking":1}
```

> En el plan gratis, el servicio "se duerme" tras 15 min sin uso; la primera petición tarda ~1 min en despertarlo.

**Alternativa:** Hugging Face Spaces → New Space → SDK *Docker* → sube el repo (usa el `Dockerfile`).

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/model-info` | Métricas, importancia de variables, reglas de prioridad |
| POST | `/predict` | Puntúa un lead → probabilidad, score, prioridad, acción |
| POST | `/predict/batch` | Puntúa una lista de leads y la devuelve ordenada (ranking) |

## 4. Desplegar el dashboard (Streamlit Community Cloud, gratis)

1. Entra a [share.streamlit.io](https://share.streamlit.io) con GitHub → **Create app**.
2. Repo: el tuyo · Branch: `main` · Main file: `dashboard/app.py`.
3. En **Advanced settings**, elige Python **3.12** → **Deploy**.

El dashboard tiene: KPIs por nivel, ranking filtrable y descargable, distribución de scores, conversión real por nivel,
curva de ganancias, un formulario para puntuar un lead nuevo y la opción de **subir un CSV** de leads para puntuarlos en lote.

## Decisiones técnicas

- **Data leakage:** se excluyen `Tags`, `Lead Quality`, `Lead Profile`, `Asymmetrique*` y `Last Notable Activity`,
  porque se llenan después del contacto comercial. Con ellas el AUC sube a ~0.98, pero el modelo no serviría para leads nuevos.
- **`Select` = faltante**, outliers recortados al percentil 99 e indicadores de faltante como features.
- **Features nuevas:** `log_time_on_site`, `time_per_visit`, `total_page_views`, `engagement_index`,
  `high_intent_activity`, `is_working_professional`.
- **Un solo módulo `features.py`** para el entrenamiento, la API y el dashboard, así se evita el *training-serving skew*.
- Las versiones de `scikit-learn` y `xgboost` están **fijadas**; si reentrenas con otras, actualiza `requirements.txt`.
