import os
import shutil
from model_factory import LogAITrainer
from inference_engine import LogAIPredictor
from generate_dummy_data import generate_synthetic_las
import logging

# Configurar logging para ver proceso
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LogAI-Test")

def main():
    logger.info("=== INICIO DE PRUEBA DE INTEGRACIÓN LOGAI-OPT ===")

    # 1. Generación de Datos Sintéticos
    logger.info("1. Generando Datos Sintéticos...")
    data_train_dir = "data_train"
    # Limpiar si existe para prueba limpia
    if os.path.exists(data_train_dir):
        shutil.rmtree(data_train_dir)
    
    generate_synthetic_las("Llanos", num_wells=6, output_dir=data_train_dir)
    generate_synthetic_las("VMM", num_wells=4, output_dir=data_train_dir)

    # 2. Entrenamiento (Offline)
    logger.info("2. Entrenando Modelos (Offline)...")
    trainer = LogAITrainer(models_dir="models")
    
    # Entrenar para DT y RHOB (puedes agregar NPHI u otros si generaste data para ellos)
    trainer.train_all_basins(data_train_path=data_train_dir, target_curves=['DT', 'RHOB'])

    # 3. Inferencia (Online/Real-time)
    logger.info("3. Probando Inferencia y VOI...")
    predictor = LogAIPredictor(models_dir="models")
    
    # Usamos uno de los archivos generados como "input" de prueba
    test_file = os.path.join(data_train_dir, "Llanos", "well_1.las")
    
    if not os.path.exists(test_file):
        logger.error("No se encontró archivo de prueba.")
        return

    # Ejecutar predicción
    las_out, voi_report, df_out = predictor.predict_and_explain(
        test_file, 
        basin_name="Llanos", 
        target_curves=['DT', 'RHOB']
    )

    if las_out:
        logger.info("=== REPORTE DE VOI (Value of Information) ===")
        print(voi_report)
        
        # Verificar que se crearon las curvas
        keys = las_out.keys()
        logger.info(f"Curvas en LAS de salida: {keys}")
        
        # Guardar resultado
        output_file = "output/test_result.las"
        os.makedirs("output", exist_ok=True)
        las_out.write(output_file, version=2.0)
        logger.info(f"Archivo generado: {output_file}")
    else:
        logger.error("Fallo en inferencia.")

    logger.info("=== PRUEBA FINALIZADA CON ÉXITO ===")
    logger.info("Para ejecutar el API: uvicorn api:app --reload")

if __name__ == "__main__":
    main()
