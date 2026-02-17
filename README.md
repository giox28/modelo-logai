# LogAI-Opt — Motor de Analítica Predictiva para Registros Geofísicos

Backend de IA para la reconstrucción de curvas geofísicas faltantes en archivos `.las` y optimización de adquisición de registros (Value of Information — VOI).

---

## Cómo Levantar la Aplicación (Paso a Paso)

### Prerrequisitos

- **Docker Desktop** instalado y corriendo
- **Node.js 18+** (solo si se quiere usar el frontend Angular)
- Archivos LAS de entrenamiento en `data_train/{nombre_cuenca}/`

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/giox28/modelo-logai.git
cd modelo-logai
```

### Paso 2: Preparar los Datos de Entrenamiento

Crear la carpeta `data_train/` con subcarpetas por cuenca. Cada subcarpeta debe contener archivos `.las`:

```
data_train/
├── cauca_patia/
│   ├── ANH CAUCA 10 STS.las
│   ├── ANH CAUCA 5 STS.las
│   └── ...
├── llanos/          (opcional, si se tienen datos)
└── vmm/             (opcional)
```

### Paso 3: Construir y Levantar el Backend (Docker)

```bash
docker-compose up --build -d
```

Esto:
- Construye la imagen Docker con Python 3.10, XGBoost, FastAPI
- Levanta el contenedor `logai-backend` en el **puerto 8001**
- Monta automáticamente: `models/`, `output/`, `data_train/`

**Verificar que está corriendo:**
```bash
docker logs logai-backend --tail 5
# Debe mostrar: "Uvicorn running on http://0.0.0.0:8000"
```

### Paso 4: Entrenar los Modelos

La primera vez (o cuando se actualicen datos de entrenamiento):

```bash
docker exec logai-backend python train_real.py
```

Este proceso:
1. Lee todos los LAS de `data_train/`
2. Aplica el pipeline QC (estandarización, conversión de unidades, detección de casing, filtro de pozos llave)
3. Entrena un modelo XGBoost por cada curva estándar
4. Guarda modelos en `models/{cuenca}/` y métricas en `metrics.json`
5. Tarda ~3-5 minutos por cuenca

**Verificar métricas:**
```
GET http://localhost:8001/model-metrics/cauca_patia
```

### Paso 5: Usar la API

La API está disponible en: **http://localhost:8001**

Documentación interactiva (Swagger): **http://localhost:8001/docs**

#### Endpoints Disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/process-well` | Procesar archivo LAS → curvas sintéticas + VOI |
| `POST` | `/inspect-well` | Inspeccionar curvas disponibles en un LAS |
| `GET` | `/available-models/{basin}` | Listar modelos entrenados por cuenca |
| `GET` | `/model-metrics/{basin}` | Métricas auditables (RMSE, R²) |
| `GET` | `/download/{filename}` | Descargar archivo LAS procesado |

#### Ejemplo con cURL

```bash
curl -X POST http://localhost:8001/process-well \
  -F "file=@mi_pozo.las" \
  -F "basin_name=cauca_patia" \
  -F "target_curves=DT,RHOB"
```

### Paso 6 (Opcional): Levantar el Frontend Angular

```bash
cd logai-front
npm install
npm start
```

El frontend estará en **http://localhost:4200** y se conecta automáticamente al backend en el puerto 8001.

---

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

## Pipeline QC (Quality Control)

Cada archivo LAS pasa por 4 etapas automáticas antes de entrenamiento o inferencia:

1. **Estandarización de mnemónicos**: 120+ alias → 14 curvas estándar (GR, ILD, NPHI, etc.)
2. **Conversión de unidades**: Detección automática M→FT desde el header LAS
3. **Detección de casing**: Zonas con CALI constante se marcan como NaN
4. **Filtro de Pozos Llave**: Solo pozos con ≥4 curvas estándar y ≥100 muestras válidas

## Diccionario de Curvas

| Mnemónico | Significado Petrofísico | Límites |
|---|---|---|
| GR | Litología (Arcillosidad) | 0–500 GAPI |
| ILD | Fluidos (Saturación de Agua) | 0.1–20000 Ohm.m |
| NPHI | Porosidad Total (Neutrón) | -0.05–0.6 v/v |
| DT | Porosidad / Mecánica de Rocas | 30–250 us/ft |
| RHOB | Porosidad Total (Densidad) | 1.0–3.5 g/cc |
| SP | Permeabilidad / Litología | -200–200 mV |
| CALI | Calidad de Hueco | 4–30 in |
| PEF | Mineralogía | 0–20 b/e |
| VSH | Volumen de Arcilla | — |
| PHIE | Porosidad Efectiva | — |
| SW | Saturación de Agua | — |
| PERM | Permeabilidad | — |

## Estructura del Proyecto

| Archivo | Función |
|---|---|
| `api.py` | API FastAPI — Endpoints REST |
| `inference_engine.py` | Motor de predicción + VOI |
| `model_factory.py` | Entrenamiento + métricas RMSE/R² |
| `utils.py` | Pipeline QC + diccionario de alias |
| `train_real.py` | Script de re-entrenamiento |
| `Dockerfile` | Imagen Docker (Python 3.10-slim) |
| `docker-compose.yml` | Orquestación con volúmenes |
| `requirements.txt` | Dependencias Python |
| `logai-front/` | Frontend Angular 17 |
