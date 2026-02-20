import os
import joblib
import pandas as pd
import numpy as np
import shap
import lasio
import logging
from utils import standardize_dataframe, ALIAS_DICT
from model_factory import UncertaintyLoggerModel # Import definition
import sys

# --- FIX for Pickle/Joblib: Map __main__.UncertaintyLoggerModel to the actual class ---
# Esto es necesario porque los modelos fueron entrenados ejecutando model_factory.py como __main__
import __main__
if not hasattr(__main__, "UncertaintyLoggerModel"):
    setattr(__main__, "UncertaintyLoggerModel", UncertaintyLoggerModel)
# ---------------------------------------------------------------------------------------

# Configuración de Logging
file_handler = logging.FileHandler(os.path.join("output", "debug.log"))
file_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Get root logger and add handler (since basicConfig might be ignored or already set)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.INFO) # Ensure root allows it

logger = logging.getLogger(__name__)

class GeoOptimaPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {} 
        self.metrics = {} # VOI v2.0
        self.master_curves = list(ALIAS_DICT.keys())
        
    def _load_metrics(self, basin):
        """Carga métricas auditables para VOI v2.0"""
        if basin in self.metrics: return
        
        try:
            metrics_path = os.path.join(self.models_dir, basin, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics[basin] = json.load(f)
            else:
                self.metrics[basin] = {}
        except Exception as e:
            logger.warning(f"No se pudieron cargar métricas para {basin}: {e}")
            self.metrics[basin] = {}

    def load_models(self, basin, target_curves):
        """Carga los modelos necesarios para una cuenca."""
        self._load_metrics(basin) # VOI v2.0: Cargar métricas
        
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

    def predict_and_explain(self, las_file_path, basin_name, basin_to_use=None, project_type='oil', target_curves=['DT', 'RHOB']):
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
        
        # 2. Cargar Modelos (Soporte para Cuencas Frontera / Análogas)
        basin_inference = basin_to_use if basin_to_use else basin_name
        self.load_models(basin_inference, target_curves)

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
            if target not in self.models.get(basin_inference, {}):
                voi_report[target] = {"error": f"Modelo no disponible para {target}"}
                continue
                
            model = self.models[basin_inference][target]
            
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
            
            # --- MEJORA 4: Predicción con Incertidumbre ---
            if hasattr(model, 'predict_with_uncertainty'):
                y_pred, y_p10, y_p90, banda = model.predict_with_uncertainty(X_input)
            else:
                # Fallback para modelos legacy
                y_pred = model.predict(X_input)
                # Estimación dummy de incertidumbre (10% del valor) para no romper
                y_p10 = y_pred * 0.9
                y_p90 = y_pred * 1.1
                banda = y_p90 - y_p10
            
            # Guardar en DataFrame para Frontend
            target_col_name = f"{target}_SYN"
            df_processed[target_col_name] = y_pred
            df_processed[f"{target}_P10"] = y_p10
            df_processed[f"{target}_P90"] = y_p90
            df_processed[f"{target}_UNCERTAINTY"] = banda
            
            # Inyectar en el archivo LAS
            unit = "US/FT" if target.startswith("DT") else "G/CC" if target == "RHOB" else ""
            las.append_curve(target_col_name, y_pred, unit=unit, descr=f"LogAI P50 Synthetic")
            las.append_curve(f"{target}_P10", y_p10, unit=unit, descr=f"LogAI P10 Lower Bound")
            las.append_curve(f"{target}_P90", y_p90, unit=unit, descr=f"LogAI P90 Upper Bound")
            
            # Metadatos ANH al Header del LAS
            las.well['LOGAI_MODEL'] = lasio.HeaderItem('LOGAI', '', 'GeoOptima - Predictiva Acotada BIEN')

            # --- MEJORA 5: Explicabilidad con SHAP ---
            try:
                # Usamos solo el modelo P50 para explicar (si es wrapper) o el modelo directo
                model_to_explain = model.model_p50 if hasattr(model, 'model_p50') else model
                
                explainer = shap.TreeExplainer(model_to_explain)
                # Tomamos una muestra (100 puntos) para no saturar la memoria si el pozo es muy largo
                X_sample = X_input.fillna(0).sample(min(100, len(X_input)), random_state=42)
                shap_values = explainer.shap_values(X_sample)
                
                # Promedio absoluto de SHAP por feature (Impacto Global en este pozo)
                mean_shap = np.abs(shap_values).mean(axis=0)
                
                # Mapear importancia a la curva base
                curve_importance = {}
                for idx, feat_name in enumerate(expected_features):
                    base = feat_name.split('_')[0]
                    if base not in curve_importance: curve_importance[base] = 0.0
                    if idx < len(mean_shap):
                        curve_importance[base] += mean_shap[idx]
                
                # Normalizar a 100%
                total_shap = sum(curve_importance.values())
                if total_shap > 0:
                    curve_importance = {k: v/total_shap for k,v in curve_importance.items()}
                
                # Extraer la incertidumbre promedio del pozo
                mean_uncertainty = float(np.nanmean(banda))
                
                # Generar reporte VOI Acotado
                suggestion, level = self._generate_acquisition_suggestion(
                    target, curve_importance, basin_inference, project_type, mean_uncertainty
                )
                
                voi_report[target] = {
                    "importance": curve_importance,
                    "suggestion": suggestion,
                    "risk_level": level,  # "AHORRO", "PRECISION", "OBLIGATORIO"
                    "mean_uncertainty": mean_uncertainty,
                    "model_features_used": list(expected_features),
                    "missing_features": missing_feats_count,
                    "shap_enabled": True
                }
                
                self.plot_shap(basin_name, target, curve_importance, os.path.basename(las_file_path))

            except Exception as e:
                logger.error(f"Error SHAP en {target}: {e}")
                voi_report[target] = {"error": f"Error SHAP: {str(e)}"}

        return las, voi_report, df_processed

    def _generate_acquisition_suggestion(self, target, importance, basin, project_type, mean_uncertainty):
        from utils import CURVE_MEANING, TOLERANCE_DICT
        ctx = {'value_term': 'Reservas', 'risk_term': 'Caracterización', 'criticality': 'Riesgo Geológico'} # Resume tu diccionario aquí
        
        target_name = CURVE_MEANING.get(target, target)
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_feat, score = sorted_feats[0] if sorted_feats else ("N/A", 0)
        
        tolerance = TOLERANCE_DICT.get(target, 1.0)
        
        # Evaluamos usando la incertidumbre real de este pozo (P90 - P10)
        is_accurate = mean_uncertainty < tolerance

        if score >= 0.40 and is_accurate:
            level = "AHORRO"
            body = f"Ahorro Confirmado. Incertidumbre (+/- {mean_uncertainty:.2f}) DENTRO de tolerancia ({tolerance}). ELIMINE HERRAMIENTA."
        elif score >= 0.40 and not is_accurate:
            level = "PRECISION"
            body = f"Riesgo de Precisión. Correlación alta ({score:.1%}), pero Incertidumbre ({mean_uncertainty:.2f}) EXCEDE tolerancia ({tolerance}). ADQUIERA HERRAMIENTA."
        else:
            level = "OBLIGATORIO"
            body = f"Riesgo Geológico. Correlación baja ({score:.1%}). SE REQUIERE ADQUISICIÓN FÍSICA OBLIGATORIA."

        return body, level

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
