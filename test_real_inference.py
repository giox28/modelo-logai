from inference_engine import LogAIPredictor
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestInference")

def test_inference_real():
    predictor = LogAIPredictor(models_dir="models")
    
    # Archivo solicitado por el usuario
    test_file = os.path.join("data_input", "EL MIEDO 2.las")
    basin = "guajira" # Asumimos guajira ya que es data real, o lo que corresponda
    
    if not os.path.exists(test_file):
        print(f"Archivo no encontrado: {test_file}")
        return
    
    print(f"Probando inferencia en {test_file}...")
    las_out, voi, df = predictor.predict_and_explain(test_file, basin, target_curves=['DT', 'RHOB'])
    
    if las_out:
        print("Inferencia exitosa.")
        print("VOI Report:", voi)
        
        base_name = os.path.splitext(os.path.basename(test_file))[0]
        output_path = f"output/{base_name}_SYN.las"
        
        las_out.write(output_path, version=2.0)
        print(f"Resultados guardados en {output_path}")
        print("Gráficas SHAP generadas en output/plots/inference/")
    else:
        print("Fallo en inferencia.")

if __name__ == "__main__":
    test_inference_real()
