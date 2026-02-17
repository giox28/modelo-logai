from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os
import uuid
import logging
from typing import List, Optional
import pandas as pd
import lasio
import numpy as np
from utils import standardize_dataframe, ALIAS_DICT, get_std_name

from inference_engine import LogAIPredictor

# Configurar Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogAI-API")

app = FastAPI(title="LogAI-Opt API", version="1.0")

# DEBUG STARTUP
print("--------------------------------------------------")
print("   API STARTING - DEBUG VERSION - CLEANING CHECK  ")
print("--------------------------------------------------")

# Configurar CORS para integración con Angular/React
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción cambiar a dominios específicos de la ANH
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios de trabajo
UPLOAD_DIR = "data_input"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instancia global del predictor (carga lazy de modelos)
predictor = LogAIPredictor()

def subsample_curve(curve_data, max_points=1000):
    """Subsamplea la data para visualización rápida en frontend."""
    # Convertir a numpy, reemplazar nulos y volver a lista
    data = np.array(curve_data)
    # Reemplazar -999.25 (valor nulo LAS estándar) con NaN
    data = np.where(np.isclose(data, -999.25), np.nan, data)
    
    if len(data) > max_points:
        indices = np.linspace(0, len(data) - 1, max_points).astype(int)
        data = data[indices]
    
    # Convertir a lista de python (float nativo, no numpy)
    return [float(x) if not np.isnan(x) else None for x in data]

def replace_nan(data):
    """Reemplaza NaN con None para JSON válido."""
    # Esta función ya no es estrictamente necesaria si subsample_curve devuelve None
    return data

def serialize_voi(report):
    """Convierte tipos de numpy en VOI report a tipos nativos."""
    if isinstance(report, dict):
        return {k: serialize_voi(v) for k, v in report.items()}
    elif isinstance(report, (list, tuple)):
        return [serialize_voi(x) for x in report]
    elif isinstance(report, (np.float32, np.float64)):
        return float(report)
    elif isinstance(report, (np.int32, np.int64)):
        return int(report)
    return report

@app.get("/available-models/{basin}")
def get_available_models(basin: str):
    """Devuelve la lista de modelos entrenados disponibles para una cuenca."""
    models_dir = os.path.join("models", basin)
    if not os.path.exists(models_dir):
        return {"models": []}
    
    # Buscar archivos .joblib
    files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]
    # El nombre del archivo es la curva (ej: DT.joblib -> DT)
    models = [os.path.splitext(f)[0] for f in files]
    return {"models": models}

@app.get("/model-metrics/{basin}")
def get_model_metrics(basin: str):
    """Devuelve métricas auditables (RMSE, R²) de los modelos entrenados para una cuenca."""
    import json
    metrics_path = os.path.join("models", basin, "metrics.json")
    if not os.path.exists(metrics_path):
        return {"basin": basin, "metrics": {}, "message": "No hay métricas disponibles. Reentrene los modelos."}
    
    with open(metrics_path, 'r') as mf:
        metrics = json.load(mf)
    
    return {"basin": basin, "metrics": metrics}

@app.post("/inspect-well")
async def inspect_well(file: UploadFile = File(...)):
    """Devuelve las curvas disponibles y data sampleada para preview."""
    try:
        content = await file.read()
        # Guardar temporalmente para leer con lasio (requiere path o stringio)
        temp_path = f"temp_{uuid.uuid4()}.las"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        try:
            las = lasio.read(temp_path)
            # Reemplazar nulos de LASIO automáticamente si no lo hizo
            df = las.df()
            df = df.replace(-999.25, np.nan)
            df = df.reset_index() # Depth como columna
            
            curves = {}
            # Estandarizar profundidad si existe
            depth_col = 'DEPT' if 'DEPT' in df.columns else df.columns[0]
            
            # Subsamplear Depth para Eje Y
            depth_data = subsample_curve(df[depth_col])
            
            for col in df.columns:
                if col == depth_col: continue
                
                # Intentar estandarizar el nombre (ej. GR_EDIT -> GR) para que coincida con los modelos
                std_name = get_std_name(col)
                final_name = std_name if std_name else col
                
                # Subsamplear curva
                curves[final_name] = subsample_curve(df[col])
            
            return JSONResponse({
                "depth": depth_data,
                "curves": curves
            })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Error inspeccionando LAS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-well")
async def process_well(
    file: UploadFile = File(...),
    basin_name: str = Form(...),
    target_curves: str = Form("DT,RHOB") # Comma separated list
):
    """
    Recibe un archivo LAS y una cuenca. Devuelve JSON con VOI y link de descarga.
    """
    try:
        # 1. Guardar archivo subido
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Archivo recibido: {filename} para cuenca {basin_name}")

        # 2. Parsear curvas objetivo
        targets = [t.strip().upper() for t in target_curves.split(',')]

        # 3. Procesar con LogAIPredictor
        # Nota: Esto es bloqueante. En prod usaríamos BackgroundTasks o Celery.
        las_out, voi_report, df_out = predictor.predict_and_explain(
            file_path, 
            basin_name, 
            target_curves=targets
        )

        if las_out is None:
            raise HTTPException(status_code=500, detail="Error procesando el archivo LAS. Formato inválido o corrupto.")
        
        if "error" in voi_report:
             raise HTTPException(status_code=400, detail=f"Error de modelo: {voi_report['error']}")

        # 4. Guardar resultado
        output_filename = f"PROCESSED_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        las_out.write(output_path, version=2.0)
        
        # 5. Respuesta
        return JSONResponse(content={
            "status": "success",
            "file_id": file_id,
            "basin": basin_name,
            "processed_file": output_filename,
            "download_url": f"/download/{output_filename}",
            "voi_report": serialize_voi(voi_report),
            "synthetic_data": {
               target: subsample_curve(df_out[f"{target}_SYN"]) 
               for target in targets 
               if f"{target}_SYN" in df_out.columns
            },
            "depth_data": subsample_curve(df_out.iloc[:, 0]) # Asumimos Depth en col 0
        })

    except Exception as e:
        logger.error(f"Error en endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type='application/octet-stream')
    else:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

# Health check
@app.get("/")
def read_root():
    return {"status": "ok", "service": "LogAI-Opt API", "version": "1.0", "docs": "/docs"}
