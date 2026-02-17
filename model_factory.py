import os
import glob
import lasio
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from utils import standardize_dataframe, ALIAS_DICT, convert_depth_units, detect_casing

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogAITrainer:
    def __init__(self, models_dir="models", output_dir="output"):
        """
        Inicializa el entrenador.
        :param models_dir: Directorio donde se guardarán los modelos entrenados.
        :param output_dir: Directorio donde se guardarán reportes y gráficas.
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Curvas Maestras (Standard Names)
        self.master_curves = list(ALIAS_DICT.keys())
        
        # Targets por defecto (intentaremos entrenar todo lo posible)
        self.default_targets = self.master_curves # ['GR', 'ILD', 'NPHI', 'DT', 'RHOB', etc]

    def load_and_process_data(self, basin_path):
        """
        Carga todos los LAS de una cuenca y genera un DataFrame consolidado y ESTANDARIZADO.
        Aplica filtros de Pozos Llave, conversión de unidades y detección de casing.
        """
        if not os.path.exists(basin_path):
            logger.warning(f"Ruta no encontrada: {basin_path}")
            return pd.DataFrame()

        las_files = glob.glob(os.path.join(basin_path, "*.las"))
        all_data = []
        
        # Constantes para filtro de Pozos Llave
        MIN_STD_CURVES = 4   # Mínimo de curvas estándar para considerar "Pozo Llave"
        MIN_VALID_SAMPLES = 100  # Mínimo de muestras válidas (no NaN)
        wells_accepted = 0
        wells_rejected = 0

        for filepath in las_files:
            try:
                las = lasio.read(filepath)
                df = las.df().reset_index() # Depth pasa a columna
                
                # --- QC PASO 1: Estandarizar nombres y limpiar datos ---
                df = standardize_dataframe(df)
                
                # --- QC PASO 2: Conversión de unidades M -> FT ---
                df, unit_status = convert_depth_units(df, las)

                # --- QC PASO 3: Detección de tramos de casing ---
                df, n_casing = detect_casing(df)

                # --- QC PASO 4: Filtro de Pozos Llave ---
                # Contar curvas estándar presentes con datos válidos
                std_curves_present = [c for c in self.master_curves 
                                      if c in df.columns and df[c].notna().sum() > 0]
                n_std_curves = len(std_curves_present)
                
                # Contar muestras válidas (filas con al menos 1 curva no-NaN)
                petro_cols = [c for c in df.columns if c in self.master_curves]
                if petro_cols:
                    n_valid = df[petro_cols].dropna(how='all').shape[0]
                else:
                    n_valid = 0
                
                well_name = os.path.basename(filepath)
                
                if n_std_curves < MIN_STD_CURVES:
                    wells_rejected += 1
                    logger.info(f"QC_KEY_WELL: DESCARTADO {well_name} | Solo {n_std_curves} curvas estándar ({std_curves_present}). Mínimo: {MIN_STD_CURVES}")
                    continue
                
                if n_valid < MIN_VALID_SAMPLES:
                    wells_rejected += 1
                    logger.info(f"QC_KEY_WELL: DESCARTADO {well_name} | Solo {n_valid} muestras válidas. Mínimo: {MIN_VALID_SAMPLES}")
                    continue

                # Extraer UWI
                uwi = las.well.UWI.value if 'UWI' in las.well else well_name
                if not uwi: uwi = well_name
                df['UWI'] = str(uwi)

                all_data.append(df)
                wells_accepted += 1
                logger.info(f"QC_KEY_WELL: ACEPTADO {well_name} | {n_std_curves} curvas, {n_valid} muestras, {n_casing} casing, unidad={unit_status}")

            except Exception as e:
                logger.error(f"Error leyendo {filepath}: {e}")

        if not all_data:
            return pd.DataFrame()

        master_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Data consolidada cuenca: {master_df.shape}. Pozos aceptados: {wells_accepted}, descartados: {wells_rejected}")
        return master_df

    def feature_engineering(self, df):
        """Genera features derivadas solo para curvas maestras presentes."""
        if 'DEPT' in df.columns:
            df = df.sort_values(by=['UWI', 'DEPT'])
        
        # Trabajar solo con curvas estandarizadas que existan en el DF
        valid_cols = [c for c in self.master_curves if c in df.columns]
        
        dfs_processed = []
        for uwi, sub_df in df.groupby('UWI'):
            sub_df = sub_df.copy()
            for col in valid_cols:
                # Gradiente y Ventana (Geoscientific Logic: tendencias locales)
                sub_df[f'{col}_GRAD'] = sub_df[col].diff()
                sub_df[f'{col}_SMOOTH'] = sub_df[col].rolling(window=10, center=True).mean()
            
            dfs_processed.append(sub_df)
        
        if not dfs_processed:
            return df
            
        return pd.concat(dfs_processed)

    def save_plots(self, basin, target, y_true, y_pred, model, features):
        """Genera y guarda gráficos de entrenamiento."""
        plot_dir = os.path.join(self.output_dir, "plots", basin)
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Pred vs Actual
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.title(f'{basin} - {target}: Real vs Predicción')
        plt.savefig(os.path.join(plot_dir, f'{target}_crossplot.png'))
        plt.close()

        # 2. Feature Importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            importance = model.feature_importances_
            # Solo plotear top 15 features
            indices = np.argsort(importance)[-15:]
            plt.barh(range(len(indices)), importance[indices], align='center')
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.title(f'{basin} - {target}: Top Feature Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{target}_importance.png'))
            plt.close()

    def train_all_basins(self, data_train_path="data_train", target_curves=None):
        targets = target_curves if target_curves else self.default_targets
        basins = [d for d in os.listdir(data_train_path) if os.path.isdir(os.path.join(data_train_path, d))]
        
        for basin in basins:
            logger.info(f"=== Procesando Cuenca: {basin} ===")
            basin_full_path = os.path.join(data_train_path, basin)
            
            # 1. Carga Inteligente (con Utils)
            df = self.load_and_process_data(basin_full_path)
            if df.empty: continue

            # 2. Feature Engineering Dinámico
            df = self.feature_engineering(df)
            
            basin_model_dir = os.path.join(self.models_dir, basin)
            os.makedirs(basin_model_dir, exist_ok=True)

            # 3. Entrenamiento Dinámico (All-vs-One)
            available_std_curves = [c for c in self.master_curves if c in df.columns]
            
            # Entrenamos para CADA curva disponible si es solicitada como target
            for target in targets:
                if target not in available_std_curves:
                    # logger.warning(f"Target {target} no data in {basin}.")
                    continue
                
                # FEATURES = Todo lo demás disponbile
                input_base_vars = [c for c in available_std_curves if c != target]
                if not input_base_vars: continue

                # Expandir con derivadas
                final_features = []
                for v in input_base_vars:
                    final_features.append(v)
                    if f'{v}_GRAD' in df.columns: final_features.append(f'{v}_GRAD')
                    if f'{v}_SMOOTH' in df.columns: final_features.append(f'{v}_SMOOTH')
                
                # Filtrar nulos en Target (Data Logic)
                # XGBoost maneja nulos en Features, pero necesitamos Target válido
                train_data = df[final_features + [target, 'UWI']].dropna(subset=[target])
                
                if len(train_data) < 50: # Mínimo historial
                    continue
                
                X = train_data[final_features]
                y = train_data[target]
                groups = train_data['UWI']
                
                logger.info(f"Entrenando {target} en {basin} usando {len(input_base_vars)} curvas base: {input_base_vars}")
                
                model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
                
                # --- Cross-Validation para métricas auditables ---
                n_unique_groups = groups.nunique()
                if n_unique_groups >= 3:
                    n_splits = min(5, n_unique_groups)
                    gkf = GroupKFold(n_splits=n_splits)
                    cv_rmse_list = []
                    cv_r2_list = []
                    for train_idx, val_idx in gkf.split(X, y, groups):
                        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
                        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
                        temp_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
                        temp_model.fit(X_train_cv, y_train_cv)
                        y_pred_cv = temp_model.predict(X_val_cv)
                        cv_rmse_list.append(float(np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))))
                        cv_r2_list.append(float(r2_score(y_val_cv, y_pred_cv)))
                    
                    avg_rmse = float(np.mean(cv_rmse_list))
                    avg_r2 = float(np.mean(cv_r2_list))
                else:
                    # Fallback: train metrics (no enough groups for CV)
                    avg_rmse = None
                    avg_r2 = None
                
                # Fit final model on all data
                model.fit(X, y)
                y_pred_full = model.predict(X)
                train_rmse = float(np.sqrt(mean_squared_error(y, y_pred_full)))
                train_r2 = float(r2_score(y, y_pred_full))
                
                # Guardar modelo
                save_path = os.path.join(basin_model_dir, f"{target}.joblib")
                joblib.dump(model, save_path)
                
                # --- Guardar métricas auditables en JSON ---
                import json
                metrics_path = os.path.join(basin_model_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as mf:
                        all_metrics = json.load(mf)
                else:
                    all_metrics = {}
                
                all_metrics[target] = {
                    "train_rmse": round(train_rmse, 4),
                    "train_r2": round(train_r2, 4),
                    "cv_rmse": round(avg_rmse, 4) if avg_rmse is not None else None,
                    "cv_r2": round(avg_r2, 4) if avg_r2 is not None else None,
                    "n_samples": len(train_data),
                    "n_features": len(final_features),
                    "n_wells": int(n_unique_groups),
                    "features_used": input_base_vars
                }
                
                with open(metrics_path, 'w') as mf:
                    json.dump(all_metrics, mf, indent=2)
                
                # Plots de validación
                self.save_plots(basin, target, y, y_pred_full, model, final_features)
                
                metric_str = f"Train RMSE={train_rmse:.4f}, R²={train_r2:.4f}"
                if avg_rmse is not None:
                    metric_str += f" | CV RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}"
                logger.info(f"Modelo guardado: {target} | {metric_str}")
