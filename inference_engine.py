import os
import joblib
import pandas as pd
import numpy as np
import shap
import lasio
import logging
from utils import standardize_dataframe, ALIAS_DICT

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
            
            # Predecir
            pred = model.predict(X_input)
            
            # Guardar
            target_col_name = f"{target}_SYN"
            df_processed[target_col_name] = pred
            
            unit = "US/FT" if target.startswith("DT") else ""
            las.append_curve(target_col_name, pred, unit=unit, descr=f"{target} Synthetic LogAI")

            # --- Optimización de Adquisición (Feature Importance Nativa) ---
            try:
                with open(os.path.join("output", "debug.txt"), "a", encoding="utf-8") as f:
                    f.write(f"\n[{target}] Calculating feature importance (native XGBoost)...\n")

                # Use XGBoost's built-in feature importance (no SHAP dependency)
                global_imp = model.feature_importances_

                with open(os.path.join("output", "debug.txt"), "a", encoding="utf-8") as f:
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

                # Generar reporte de sugerencia (Context Aware)
                suggestion = self._generate_acquisition_suggestion(target, curve_importance, basin_inference, project_type)
                
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

    def _generate_acquisition_suggestion(self, target, importance, basin, project_type='oil'):
        """
        Genera una recomendación de negocio basada en RIESGO y COSTO-EFICIENCIA.
        Transforma el 'feature importance' en una decisión de adquisición de herramienta.
        Adapta la terminología según el tipo de proyecto (H2, Geotermia, CCUS).
        """
        from utils import CURVE_MEANING, TOLERANCE_DICT
        
        # Diccionario de Contexto para Nuevas Energías
        PROJECT_CONTEXT = {
            'oil': {
                'value_term': 'Reservas Probadas',
                'risk_term': 'Caracterización de Yacimiento',
                'criticality': 'Reducción de Incertidumbre'
            },
            'geothermal': {
                'value_term': 'Capacidad de Generación',
                'risk_term': 'Riesgo Entálpico / Flujo de Calor',
                'criticality': 'Identificación de Zonas Productivas'
            },
            'hydrogen': {
                'value_term': 'Potencial de Generación (H2 Natural/Blanco)',
                'risk_term': 'Riesgo de Migración y Trampa',
                'criticality': 'Detección de Anomalías de Gas'
            },
            'ccus': {
                'value_term': 'Capacidad de Inyección',
                'risk_term': 'Riesgo de Fuga (Leakage)',
                'criticality': 'Monitoreo de Pluma de CO2'
            }
        }
        
        ctx = PROJECT_CONTEXT.get(project_type, PROJECT_CONTEXT['oil'])

        # 1. Validar si hay predictores
        if not importance:
            return f"⚠️ DATA INSUFICIENTE: No se pudo determinar una correlación fiable. SE REQUIERE ADQUISICIÓN FÍSICA para garantizar {ctx['risk_term']}."

        # 2. Identificar el predictor dominante
        sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_feat, score = sorted_feats[0]
        
        target_name = CURVE_MEANING.get(target, target)
        input_name = CURVE_MEANING.get(top_feat, top_feat)
        
        # Obtener métricas de precisión (RMSE) del modelo
        basin_metrics = self.metrics.get(basin, {})
        target_metrics = basin_metrics.get(target, {})
        cv_rmse = target_metrics.get('cv_rmse', 999.0)
        tolerance = TOLERANCE_DICT.get(target, 1.0)
        
        is_accurate = cv_rmse < tolerance

        # 3. Lógica Híbrida: Semáforo de Decisión (Negocio + RMSE)
        
        # CASO A: ALTA CONFIANZA y BAJO ERROR (Ahorro Seguro)
        if score >= 0.50 and is_accurate:
            title = f"💡 OPORTUNIDAD DE OPTIMIZACIÓN (Ahorro Confirmado - {ctx['value_term']})"
            body = (
                f"El análisis indica una redundancia petrofísica alta. La herramienta **{target_name} ({target})** "
                f"tiene una dependencia del **{score:.1%}** con **{input_name} ({top_feat})**.\n\n"
                f"✅ **Estrategia Recomendada:** En pozos de desarrollo, considere **ELIMINAR** la corrida de {target} "
                f"y reconstruirla virtualmente. El error esperado (+/- {cv_rmse:.2f}) está DENTRO de la tolerancia operativa ({tolerance})."
            )

        # CASO B: ALTA CONFIANZA pero ALTO ERROR (Falso Positivo - Riesgo de Precisión)
        elif score >= 0.50 and not is_accurate:
            title = f"⚠️ RIESGO DE PRECISIÓN (Adquisición Recomendada para {ctx['risk_term']})"
            body = (
                f"Aunque existe una fuerte correlación estadística ({score:.1%}), el modelo presenta un error físico de "
                f"**{cv_rmse:.2f}**, que EXCEDE la tolerancia permitida ({tolerance}).\n\n"
                f"⚠️ **Acción:** No elimine la herramienta. La reconstrucción virtual podría enmascarar detalles críticos para {ctx['criticality']}."
            )

        # CASO C: BAJA CONFIANZA (Riesgo Geológico/Físico)
        elif score < 0.30:
            title = f"⛔ ALERTA DE {ctx['risk_term'].upper()} (Adquisición Obligatoria)"
            body = (
                f"El modelo NO puede reconstruir **{target_name} ({target})** con fiabilidad. "
                f"La mejor correlación solo explica el **{score:.1%}** de la variabilidad.\n\n"
                f"⛔ **Acción Crítica:** **SE REQUIERE ADQUISICIÓN FÍSICA**. Mantenga esta herramienta en el programa para "
                f"garantizar la seguridad y viabilidad del proyecto de {project_type.upper()}."
            )

        # CASO D: ZONA GRIS (Requiere Calibración)
        else:
            title = "🔍 OPORTUNIDAD CONDICIONAL (Requiere Calibración)"
            body = (
                f"Existe una correlación moderada (**{score:.1%}**) entre **{target}** y **{top_feat}**. "
                f"El error del modelo es **{cv_rmse:.2f}** (Tol: {tolerance}).\n\n"
                f"⚠️ **Estrategia:** El reemplazo es viable PERO requiere validación con núcleos o pozos vecinos (offset wells)."
            )

        return f"### {title}\n\n{body}"

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
