# Lead Scoring Predictivo — X Education

Modelo predictivo que clasifica leads según su probabilidad de conversión. Está desplegado como un **sistema funcional**
con ranking de leads, sistema de priorización y dashboard comercial.

**🔗 Dashboard desplegado:** https://lead-scoring-kevin.streamlit.app
**📓 Notebook (Colab):** [`Lead_Scoring_Predictivo.ipynb`](Lead_Scoring_Predictivo.ipynb)
**📊 Dataset:** [Lead Scoring Case Study — Kaggle](https://www.kaggle.com/code/kratikulhara/lead-scoring-case-study) (9.240 leads, 37 columnas)

## Resumen de resultados

| Indicador | Resultado |
|---|---|
| Mejor modelo | **XGBoost** con hiperparámetros optimizados (RandomizedSearchCV) |
| ROC-AUC | **0.907** validación cruzada · **0.890** test · **0.905** fuera de muestra (todo el dataset) |
| Modelos comparados | Regresión logística, Gradient Boosting, XGBoost, Red neuronal (MLP) |
| Priorización | 🔥 Caliente convierte **86%** · 🌤 Tibio **58%** · ❄️ Frío **12%** (promedio general: 38.5%) |
| Impacto comercial | Contactando solo el **30%** superior del ranking se captura el **66%** de las conversiones |

## Cumplimiento de requerimientos

| Requerimiento | Dónde se evidencia |
|---|---|
| Modelos de clasificación | Sección 4 · notebook §5 |
| Feature engineering | Sección 3 · `features.py` · notebook §4 |
| Gradient Boosting y XGBoost | Sección 4 · notebook §5 |
| Redes neuronales artificiales | Sección 4 (MLP 64→32) · notebook §5 |
| Evaluación ROC-AUC | Sección 4 · notebook §6 |
| Ranking de leads | Sección 6 · dashboard, pestaña *Ranking* |
| Sistema de priorización | Sección 6 · dashboard, KPIs, filtros y pestaña *Puntuar un lead* |
| Dashboard comercial desplegado | Sección 7 · https://lead-scoring-kevin.streamlit.app |
| Endpoint | Sección 8 · `api/main.py` (FastAPI) |

---

## 1. Análisis exploratorio (EDA)

En este dataset, el valor `Select` significa que el usuario no eligió ninguna opción en el formulario, así que se trata
como **valor faltante**. Varias columnas superan el 30–70% de faltantes.

![Valores faltantes](docs/analisis/01_valores_faltantes.png)

![Conversión por categoría](docs/analisis/02_conversion_por_categoria.png)

![Variables numéricas](docs/analisis/03_variables_numericas.png)

**Hallazgos**
- La conversión global es del **38.5%**, así que las clases están moderadamente desbalanceadas.
- **Tiempo en el sitio** es la variable numérica que más separa: quienes convierten pasan mucho más tiempo navegando.
- Los leads de `Lead Add Form` (92%) y los **Working Professionals** (92%) convierten muy por encima del promedio.
- `TotalVisits` y `Page Views Per Visit` tienen outliers extremos (máximos de 251 y 55), así que se recortan al percentil 99.

## 2. Fuga de información (data leakage)

Columnas como `Tags`, `Lead Quality`, `Lead Profile`, `Asymmetrique*` y `Last Notable Activity` las registra el equipo
comercial **después** de trabajar el lead. Por ejemplo, `Tags = "Closed by Horizzon"` convierte el 99%.

Con esas columnas el ROC-AUC sube a **~0.98**, pero el modelo no serviría para puntuar un lead nuevo, porque al llegar
el lead todavía no tiene esos datos. **Por eso se excluyen.** El modelo usa solo información disponible en el momento
en que entra el lead.

## 3. Feature engineering

Todas las transformaciones están en [`features.py`](features.py). El mismo módulo lo usan el notebook, la API y el
dashboard, así el modelo recibe exactamente el mismo preprocesamiento en entrenamiento y en producción.

| Feature | Descripción |
|---|---|
| `log_time_on_site` | log(1 + tiempo en el sitio), para suavizar la asimetría |
| `time_per_visit` | tiempo / (visitas + 1): qué tan profunda es cada visita |
| `total_page_views` | visitas × páginas por visita |
| `engagement_index` | promedio normalizado de tiempo, visitas y páginas |
| `high_intent_activity` | la última actividad es SMS, llamada, clic en un email o formulario enviado |
| `is_working_professional` | la ocupación es Working Professional |
| `*_missing` | indicadores de faltante: no responder también es información |
| `is_india`, `do_not_email`, `free_copy` | variables binarias |
| Categóricas | One-hot, con agrupación automática de categorías raras (menos de 30 casos) |

Además, se imputan las numéricas con la mediana, se recortan outliers y se unifican los nombres de las fuentes
(`google` → `Google`, `Facebook` → `Social Media`, etc.).

## 4. Modelos y evaluación ROC-AUC

Se usó una partición estratificada 80/20 y validación cruzada de 5 folds sobre el conjunto de entrenamiento.

| Modelo | ROC-AUC CV | ROC-AUC Test | PR-AUC Test | F1 Test (umbral 0.5) |
|---|---|---|---|---|
| **XGBoost (tuned)** | **0.9072** ± 0.003 | 0.8902 | **0.8346** | **0.773** |
| Gradient Boosting | 0.9053 ± 0.003 | **0.8904** | 0.8325 | 0.767 |
| Red Neuronal (MLP 64→32, ReLU, Adam) | 0.9034 ± 0.003 | 0.8777 | 0.8131 | 0.742 |
| Regresión Logística (línea base) | 0.8997 ± 0.004 | 0.8779 | 0.8148 | 0.735 |

![Curvas ROC](docs/analisis/04_curvas_roc.png)

Se eligió **XGBoost** por tener el mejor ROC-AUC en validación cruzada, que es la estimación más estable.
Con el umbral óptimo de Youden (0.33), el modelo logra un recall del **87%** sobre los leads que convierten,
con una precisión del 72% y una exactitud del 82%.

![Matriz de confusión](docs/analisis/05_matriz_confusion.png)

**¿Por qué los modelos quedan tan cerca?** La señal está concentrada en pocas variables fuertes (fuente, origen,
ocupación y última actividad), así que hasta la regresión logística captura gran parte. El boosting aporta
interacciones no lineales que suben el AUC entre 1 y 1.5 puntos.

## 5. Variables más importantes

![Importancia de variables](docs/analisis/06_importancia_variables.png)

Las variables con más peso son la fuente **Reference**, el origen **Lead Add Form**, la **ocupación** (incluido el hecho
de no haberla respondido), la última actividad **SMS Sent** y el **tiempo en el sitio**.

## 6. Ranking y sistema de priorización

La probabilidad del modelo se convierte en un **score de 0 a 100**. Los leads se ordenan por score (ranking) y cada uno
se asigna a un nivel con una acción comercial concreta. Las cifras de la tabla son fuera de muestra, sobre los 9.240 leads.

| Nivel | Score | Acción sugerida | % de leads | Conversión real | % de conversiones capturadas |
|---|---|---|---|---|---|
| 🔥 Caliente | ≥ 70 | Llamar hoy: asignar a un asesor senior | 26% | **86%** | 59% |
| 🌤 Tibio | 40 – 69 | Contactar esta semana: llamada + email personalizado | 15% | 58% | 22% |
| ❄️ Frío | < 40 | Nurturing automático: secuencia de emails / SMS | 59% | 12% | 19% |

![Ganancias por decil](docs/analisis/07_ganancias_por_decil.png)

El 10% superior del ranking convierte el **97%**. Contactando solo el 30% superior se captura el **66%** de las
conversiones, frente al 30% que se capturaría eligiendo leads al azar.

## 7. Sistema desplegado — evidencias

Dashboard comercial en Streamlit Community Cloud: **https://lead-scoring-kevin.streamlit.app**

**Inicio:** KPIs por nivel de prioridad y ranking de leads con su score, prioridad y acción sugerida.
![Inicio y ranking](docs/evidencias/01_inicio_kpis_ranking.jpg)

**Sistema de priorización con filtros:** vista filtrada solo a leads 🔥 Calientes (2.436 leads, 85% de probabilidad media).
![Filtro calientes](docs/evidencias/07_filtro_solo_calientes.jpg)

**Análisis:** distribución de scores y conversión real por nivel de prioridad.
![Análisis 1](docs/evidencias/02_analisis_distribucion_conversion.jpg)

**Análisis:** score medio por fuente y curva de ganancias.
![Análisis 2](docs/evidencias/03_analisis_fuentes_ganancias.jpg)

**Puntuar un lead nuevo:** formulario con los datos del lead.
![Formulario](docs/evidencias/04_puntuar_lead_formulario.jpg)

**Resultado en tiempo real:** el modelo desplegado asigna score 95.8, prioridad Caliente y la acción sugerida.
![Resultado](docs/evidencias/05_puntuar_lead_resultado.jpg)

**Modelo:** comparación de los 4 modelos (ROC-AUC) e importancia de variables.
![Modelo](docs/evidencias/06_modelo_comparacion_importancia.jpg)

El dashboard también permite **subir un CSV de leads nuevos** para puntuarlos en lote y **descargar el ranking** filtrado.

## 8. Endpoint (API)

[`api/main.py`](api/main.py) expone el modelo con FastAPI (documentación interactiva en `/docs`).

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio |
| GET | `/model-info` | Métricas, importancia de variables y reglas de prioridad |
| POST | `/predict` | Puntúa un lead: probabilidad, score, prioridad y acción |
| POST | `/predict/batch` | Puntúa una lista de leads y la devuelve ordenada (ranking) |

Ejemplo de petición a `/predict`:
```json
{"lead_id": "CRM-001", "lead_origin": "Landing Page Submission", "lead_source": "Google",
 "total_visits": 5, "total_time_spent_on_website": 1200, "page_views_per_visit": 3,
 "last_activity": "SMS Sent", "current_occupation": "Working Professional", "city": "Mumbai"}
```
Respuesta:
```json
{"lead_id": "CRM-001", "probabilidad": 0.9642, "score": 96.4, "prioridad": "Caliente",
 "accion_sugerida": "Llamar hoy: asignar a un asesor senior", "ranking": 1}
```

## 9. Estructura del repositorio

```
├── Lead_Scoring_Predictivo.ipynb   # EDA, features, modelos, ROC-AUC, ranking y exportación
├── features.py                     # Limpieza, feature engineering y reglas de priorización
├── api/main.py                     # Endpoint FastAPI
├── dashboard/app.py                # Dashboard comercial Streamlit
├── models/                         # Modelo entrenado (.joblib) y metadata.json (métricas)
├── data/                           # Leads.csv (original) y leads_scored.csv (ranking)
└── docs/                           # Gráficas del análisis y evidencias del despliegue
```

## 10. Conclusiones

- Excluir las variables con fuga de información da un modelo **honesto y utilizable** para leads nuevos, aunque su AUC
  (0.89–0.91) sea menor que el de notebooks públicos que usan `Tags` (~0.98).
- **XGBoost** es el mejor modelo, pero con poca ventaja: la mayor parte de la señal viene de pocas variables de origen,
  perfil y comportamiento.
- El sistema de priorización **separa claramente** los leads (86% vs. 12% de conversión entre Caliente y Frío), lo que
  permite que el equipo comercial concentre su esfuerzo donde más rinde.

**Trabajo futuro:** calibración de probabilidades, monitoreo de drift y reentrenamiento periódico con datos reales del CRM.
