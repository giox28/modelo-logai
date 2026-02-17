# LogAI-Opt — Motor de Analítica Predictiva para Registros Geofísicos

## Descripción
Backend de IA para la reconstrucción de curvas geofísicas faltantes en archivos `.las` y optimización de adquisición de registros (Value of Information — VOI).

## Arquitectura

```
┌─────────────────┐    HTTP/REST     ┌──────────────────────────────┐
│  Angular Front  │ ◄──────────────► │   FastAPI Backend (api.py)   │
│  (logai-front/) │    :8001/4200    │                              │
└─────────────────┘                  │  ┌────────────────────────┐  │
                                     │  │ inference_engine.py    │  │
                                     │  │  - Carga modelos       │  │
                                     │  │  - Predice curvas      │  │
                                     │  │  - Calcula VOI         │  │
                                     │  └────────────────────────┘  │
                                     │  ┌────────────────────────┐  │
                                     │  │ utils.py               │  │
                                     │  │  - QC Pipeline         │  │
                                     │  │  - Alias Dict          │  │
                                     │  │  - Physical Limits     │  │
                                     │  └────────────────────────┘  │
                                     └──────────────────────────────┘
                                                  │
                                     ┌────────────┴────────────┐
                                     │  models/{basin}/*.joblib│
                                     │  models/{basin}/metrics │
                                     └─────────────────────────┘
```

## Flujo de Datos

1. **Ingesta**: Usuario sube un archivo `.las` vía API o frontend.
2. **QC Pipeline** (`utils.py`):
   - Estandarización de mnemónicos (120+ alias → 14 curvas estándar)
   - Eliminación de nulos de sistema (-999.25, -9999)
   - Filtrado por límites físicos (GR: 0–500 GAPI, RHOB: 1.0–3.5 g/cc, etc.)
   - Limpieza de cadenas malformadas (regex)
3. **Predicción** (`inference_engine.py`):
   - Carga de modelos pre-entrenados por cuenca
   - Feature engineering (gradientes, promedios móviles)
   - Predicción XGBoost para curvas faltantes
4. **VOI** (`inference_engine.py`):
   - Cálculo de importancia de features nativa (XGBoost)
   - Generación de recomendaciones de adquisición
5. **Salida**: Archivo `.las` enriquecido + reporte VOI JSON

## Estructura de Archivos

| Archivo | Función |
|---|---|
| `api.py` | API FastAPI — Endpoints REST |
| `inference_engine.py` | Motor de predicción + VOI |
| `model_factory.py` | Entrenamiento de modelos + métricas |
| `utils.py` | Pipeline QC + diccionario de alias |
| `train_real.py` | Script de re-entrenamiento |
| `docker-compose.yml` | Orquestación Docker |
| `requirements.txt` | Dependencias Python |

## Diccionario de Curvas Estándar

| Mnemónico | Significado Petrofísico | Límites Físicos |
|---|---|---|
| GR | Litología (Arcillosidad) | 0–500 GAPI |
| ILD | Fluidos (Saturación de Agua) | 0.1–20000 Ohm.m |
| NPHI | Porosidad Total (Neutrón) | -0.05–0.6 v/v |
| DT | Porosidad y Propiedades Mecánicas | 30–250 us/ft |
| RHOB | Porosidad Total (Densidad) | 1.0–3.5 g/cc |
| SP | Permeabilidad Relativa / Litología | -200–200 mV |
| CALI | Calidad de Hueco | 4–30 in |
| PEF | Litología (Mineralogía) | 0–20 b/e |
| VSH | Volumen de Arcilla | — |
| PHIE | Porosidad Efectiva | — |
| SW | Saturación de Agua | — |
| PERM | Permeabilidad | — |

## API Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/process-well` | Procesar archivo LAS, generar curvas sintéticas + VOI |
| `POST` | `/inspect-well` | Inspeccionar curvas disponibles en un LAS |
| `GET` | `/available-models/{basin}` | Listar modelos entrenados por cuenca |
| `GET` | `/model-metrics/{basin}` | Métricas auditables (RMSE, R²) por cuenca |
| `GET` | `/download/{filename}` | Descargar archivo LAS procesado |
| `GET` | `/docs` | Documentación interactiva Swagger/OpenAPI |

## Despliegue con Docker

```bash
# Construir y levantar
docker-compose up --build -d

# API disponible en: http://localhost:8001
# Swagger UI:        http://localhost:8001/docs
```

## Re-entrenamiento de Modelos

```bash
# 1. Colocar archivos LAS en data_train/{nombre_cuenca}/
# 2. Ejecutar dentro del contenedor:
docker exec logai-backend python train_real.py

# 3. Verificar métricas generadas:
# GET http://localhost:8001/model-metrics/cauca_patia
```

Las métricas se guardan automáticamente en `models/{basin}/metrics.json` con:
- `train_rmse` / `train_r2`: Métricas de ajuste (in-sample)
- `cv_rmse` / `cv_r2`: Métricas de validación cruzada (out-of-sample, GroupKFold por pozo)
- `n_samples`, `n_wells`, `n_features`: Metadata de trazabilidad

## Dependencias

- Python 3.10+
- XGBoost ≥2.0, Pandas ≥2.0, NumPy <2.0
- FastAPI, Uvicorn, Lasio, Joblib, Scikit-learn
- Matplotlib, Seaborn (visualización)

## Frontend (Angular)

```bash
cd logai-front
npm install
npm start
# Disponible en http://localhost:4200
```
