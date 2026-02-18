# GeoOptima — Motor de Analítica Predictiva para Registros Geofísicos

Backend de IA para la reconstrucción de curvas geofísicas faltantes en archivos `.las` y optimización de adquisición de registros (Value of Information — VOI). Implementa el estándar de **"Predictiva Acotada"** mediante doble verificación (Importancia + Tolerancia Física).

---

## Características Clave

- **Reconstrucción Virtual (Data Rescue):** Generación de curvas sintéticas (DT, RHOB, NPHI, etc.) usando aprendizaje automático de pozos vecinos.
- **Sugerencia de Adquisición (VOI v2.0):** Reporte de decisión financiera con tres niveles de recomendación basados en **Precisión Operativa**:
  - 💡 **AHORRO SEGURO**: La IA reconstruye con error menor a la tolerancia física. (¡Elimine la herramienta!)
  - ⚠️ **RIESGO DE PRECISIÓN**: La IA correlaciona bien, pero el error es alto. (¡Adquiera el registro!)
  - ⛔ **RIESGO GEOLÓGICO**: No hay correlación física. (¡Adquisición obligatoria!)
- **Auditoría Transparente:** Métricas `metrics.json` públicas con RMSE y R² de validación cruzada.
- **QC Geocientífico:** Pipeline automático de limpieza, conversión de unidades y detección de casing.

---

## Arquitectura

```
┌─────────────────┐    HTTP/REST     ┌──────────────────────────────┐
│  Angular Front  │ ◄──────────────► │   GeoOptima API (api.py)     │
│  (logai-front/) │    :8001/4200    │                              │
└─────────────────┘                  │  ┌────────────────────────┐  │
                                     │  │ GeoOptimaPredictor     │  │
                                     │  │ (inference_engine.py)  │  │
                                     │  │  - Carga modelos       │  │
                                     │  │  - Predice curvas      │  │
                                     │  │  - VOI v2.0 (RMSE/Tol) │  │
                                     │  └────────────────────────┘  │
                                     │  ┌────────────────────────┐  │
                                     │  │ utils.py               │  │
                                     │  │  - QC Pipeline         │  │
                                     │  │  - ALIAS_DICT          │  │
                                     │  │  - TOLERANCE_DICT      │  │
                                     │  └────────────────────────┘  │
                                     └──────────────────────────────┘
                                                  │
                                     ┌────────────┴────────────┐
                                     │  models/{basin}/*.joblib│
                                     │  models/{basin}/metrics │
                                     └─────────────────────────┘
```

## Guía de Despliegue

### Prerrequisitos
- Docker Desktop
- Node.js 18+ (Opcional, frontend)

### 1. Levantar Servicios
```bash
docker-compose up --build -d
```
Backend disponible en: **http://localhost:8001**

### 2. Entrenar Modelos
```bash
docker exec logai-backend python train_real.py
```
Este proceso genera los modelos y las métricas de precisión (`metrics.json`) necesarias para el VOI v2.0.

### 3. Usar la API
Documentación interactiva: **http://localhost:8001/docs**

---

## Predictiva Acotada: Estándares

GeoOptima cumple con los requisitos de "Predictiva Acotada":

1.  **Casos Aprobados:** Limitado a 14 curvas estándar definidas en `ALIAS_DICT`.
2.  **Métricas Publicadas:** Endpoint `/model-metrics/{basin}` expone RMSE y R² de validación cruzada.
3.  **Evidencia de Entrenamiento:** `metrics.json` incluye número de pozos y muestras usadas.
4.  **Tolerancias Físicas:** Decisiones basadas en `TOLERANCE_DICT` (ej. RHOB +/- 0.08 g/cc).

### Tolerancias Operativas (Ejemplo)

| Curva | Tolerancia (+/- RMSE) | Unidad |
|---|---|---|
| **RHOB** | 0.08 | g/cc |
| **DT** | 10.0 | us/ft |
| **NPHI** | 0.045 | v/v |
| **GR** | 15.0 | GAPI |
| **CALI** | 0.5 | in |

---

## Estructura de Archivos

| Archivo | Clase Principal | Función |
|---|---|---|
| `api.py` | `FastAPI` | API REST para GeoOptima |
| `inference_engine.py` | `GeoOptimaPredictor` | Motor de inferencia y lógica VOI v2.0 |
| `model_factory.py` | `GeoOptimaTrainer` | Entrenamiento y cálculo de métricas |
| `utils.py` | — | Diccionarios (Alias, Tolerancias) y funciones QC |
| `train_real.py` | — | Script de orquestación de entrenamiento |
