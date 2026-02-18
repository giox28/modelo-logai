import os
import shutil
from model_factory import GeoOptimaTrainer
import logging

logging.basicConfig(level=logging.INFO)

def train_real():
    # 1. Limpiar modelos previos para asegurar que entrenamos con la data nueva
    if os.path.exists("models"):
        # shutil.rmtree("models") # Retires mount point error
        for filename in os.listdir("models"):
            file_path = os.path.join("models", filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    # 2. Entrenar
    print("Iniciando entrenamiento con datos reales...")
    trainer = GeoOptimaTrainer(models_dir="models")
    # Entrenamos para todas las cuencas en data_train (incluyendo guajira)
    trainer.train_all_basins(data_train_path="data_train") 
    print("Entrenamiento finalizado.")

if __name__ == "__main__":
    train_real()
