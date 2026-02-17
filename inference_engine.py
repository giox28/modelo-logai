import os
import joblib
import pandas as pd
import numpy as np
import shap
import lasio
import logging
from utils import standardize_dataframe, ALIAS_DICT

# Configuración de Logging
file_handler = logging.FileHandler("/app/output/debug.log")
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Get root logger and add handler (since basicConfig might be ignored or already set)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO) # Ensure root allows it

logger = logging.getLogger(__name__)

class LogAIPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {} 
        self.master_curves = list(ALIAS_DICT.keys())

    def load_models(self, basin, target_curves):
        """Carga los modelos necesarios para una cuenca."""
        if basin not in self.models:
            self.models[basin] = {}
        
        loaded_count = 0
        for curve in target_curves:
            if curve in self.models[basin]:
                loaded_count += 1
                continue
            
            model_path = os.path.join(self.models_dir, basin, f"{curve}.joblib")
            if os.path.exists(model_path):
                self.models[basin][curve] = joblib.load(model_path)
                loaded_count += 1
            else:
                logger.warning(f"Modelo no encontrado: {model_path} (¿Aún no entrenado?)")
        
        return loaded_count > 0

    def predict_and_explain(self, las_file_path, basin_name, target_curves=['DT', 'RHOB']):
        """
        Retorna: las (modificado), voi_report, df_processed
        """
        try:
            las = lasio.read(las_file_path)
            df = las.df().reset_index()
        except Exception as e:
            logger.error(f"Error leyendo LAS {las_file_path}: {e}")
            return None, {}, pd.DataFrame()

        # 1. Pipeline de Limpieza (Petrophysical QC)
        logger.error("DEBUG_ENGINE: Calling standardize_dataframe...")
        df_processed = standardize_dataframe(df) # Nombres std, filtrado de outliers
        logger.error(f"DEBUG_ENGINE: standardize_dataframe returned. Columns: {df_processed.columns.tolist()}")
        logger.error(f"DEBUG_ENGINE: PEF dtype: {df_processed['PEF'].dtype if 'PEF' in df_processed.columns else 'MISSING'}")
        
        # 2. Cargar Modelos
        self.load_models(basin_name, target_curves)

        # 3. Feature Engineering (Generar GRAD y SMOOTH para todo lo disponible)
        # Esto es necesario porque el modelo puede requerir f'{algo}_GRAD'
        if 'DEPT' in df_processed.columns:
            df_processed = df_processed.sort_values(by='DEPT')
            
        for col in df_processed.columns:
            if col in self.master_curves: # Solo generar para curvas físicas conocidas
                if df_processed[col].count() > 10: # Evitar computar sobre columnas vacías
                    df_processed[f'{col}_GRAD'] = df_processed[col].diff()
                    df_processed[f'{col}_SMOOTH'] = df_processed[col].rolling(window=10, center=True).mean()

        voi_report = {}
        
        for target in target_curves:
            # Check si existe modelo
            if target not in self.models.get(basin_name, {}):
                voi_report[target] = {"error": f"Modelo no disponible para {target}"}
                continue
                
            model = self.models[basin_name][target]
            
            # --- Alineación Dinámica de Features ---
            # XGBoost guarda los nombres de features con los que entrenó
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
            else:
                try:
                    expected_features = model.get_booster().feature_names
                except:
                    logger.error(f"Modelo {target} sin metadata de features. No se puede predecir.")
                    continue
            
            # Crear DataFrame X con TODAS las columnas esperadas
            # Si falta una feature (ej. entrenamos con ILD pero este pozo no tiene), rellenar con NaN
            X_input = pd.DataFrame(index=df_processed.index)
            missing_feats_count = 0
            
            for feat in expected_features:
                if feat in df_processed.columns:
                    X_input[feat] = df_processed[feat]
                else:
                    X_input[feat] = np.nan
                    missing_feats_count += 1
            
            # FORENSIC DEBUGGING FOR SHAP INPUT
            try:
                logger.error(f"DEBUG_FORENSIC: X_input columns for target {target}: {X_input.columns.tolist()}")
                for col in X_input.columns:
                    if X_input[col].dtype == object or str(X_input[col].dtype) == 'object':
                        bad_samples = X_input[col].unique()[:3]
                        logger.error(f"DEBUG_FORENSIC: DIRTY COLUMN FOUND IN X_INPUT! Col: {col} | Samples: {bad_samples}")
                    # Also check for sneaky strings in float columns? (unlikely in pandas, but check logic)
            except Exception as e:
                logger.error(f"DEBUG_FORENSIC: Error inspecting inputs: {e}")

            # Advertencia de calidad de predicción (Acquisition logic)
            # Si faltan muchas features críticas, la predicción será mala (XGBoost usa default path)
            
            # Predecir
            pred = model.predict(X_input)
            
            # Guardar
            target_col_name = f"{target}_SYN"
            df_processed[target_col_name] = pred
            
            unit = "US/FT" if target.startswith("DT") else ""
            las.append_curve(target_col_name, pred, unit=unit, descr=f"{target} Synthetic LogAI")

            # --- Optimización de Adquisición (Feature Importance Nativa) ---
            try:
                with open("/app/output/debug.txt", "a") as f:
                    f.write(f"\n[{target}] Calculating feature importance (native XGBoost)...\n")

                # Use XGBoost's built-in feature importance (no SHAP dependency)
                global_imp = model.feature_importances_

                with open("/app/output/debug.txt", "a") as f:
                    f.write(f"[{target}] SUCCESS: Got {len(global_imp)} importance values.\n")

                # Mapear importancia a curva base (sumar GR, GR_GRAD, GR_SMOOTH)
                curve_importance = {}
                for idx, feat_name in enumerate(expected_features):
                    base = feat_name.split('_')[0]
                    if base not in curve_importance: curve_importance[base] = 0.0
                    if idx < len(global_imp):
                        curve_importance[base] += global_imp[idx]
                
                # Normalizar
                total = sum(curve_importance.values())
                if total > 0:
                    curve_importance = {k: v/total for k,v in curve_importance.items()}

                # Generar reporte de sugerencia
                suggestion = self._generate_acquisition_suggestion(target, curve_importance)
                
                voi_report[target] = {
                    "importance": curve_importance,
                    "suggestion": suggestion,
                    "model_features_used": list(expected_features),
                    "missing_features": missing_feats_count
                }
                
                self.plot_shap(basin_name, target, curve_importance, os.path.basename(las_file_path))

            except Exception as e:
                logger.error(f"Feature importance error {target}: {e}")
                voi_report[target] = {
                    "error": f"Error en análisis de impacto: {str(e)}",
                    "suggestion": "No se pudo generar recomendación debido a un error en el cálculo de impacto.",
                    "importance": {}
                }

        return las, voi_report, df_processed

    def _generate_acquisition_suggestion(self, target, importance):
        """Genera un texto de negocio justificando adquisición con criterio geocientífico."""
        from utils import CURVE_MEANING
        
        # Ordenar features por impacto
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        if not sorted_feats:
            return "No se encontraron predictores significativos. Se recomienda revisión geológica del pozo."
            
        top_feat, score = sorted_feats[0]
        top_meaning = CURVE_MEANING.get(top_feat, "Propiedad Física Desconocida")
        target_meaning = CURVE_MEANING.get(target, "Propiedad Objetivo")
        
        # Lógica de Recomendación de Negocio
        suggestion = (
            f"La herramienta más crítica para caracterizar **{target_meaning} ({target})** en esta cuenca es "
            f"**{top_feat} ({top_meaning})**.\n\n"
            f"Análisis de Impacto: La inclusión de {top_feat} explica el **{score:.1%}** de la variabilidad del modelo. "
            f"Se RECOMIENDA PRIORIZAR su adquisición en futuros programas exploratorios para minimizar la incertidumbre petrofísica."
        )
        
        return suggestion

    def plot_shap(self, basin, target, importance, filename):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            out = os.path.join("output", "plots", "inference")
            os.makedirs(out, exist_ok=True)
            plt.figure(figsize=(8,5))
            sns.barplot(x=list(importance.values()), y=list(importance.keys()))
            plt.title(f"LogAI Acquisition Optimizer\nTarget: {target} | Well: {filename}")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"VOI_{target}_{filename}.png"))
            plt.close()
        except: pass

if __name__ == "__main__":
    pass
