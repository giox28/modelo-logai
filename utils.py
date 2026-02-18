import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE CONOCIMIENTO GEOCIENTÍFICO ---

# Diccionario Maestro de Alias (Mnemónicos)
# Diccionario Maestro de Alias (Mnemónicos)
ALIAS_DICT = {
    # Gamma Ray
    'GR': ['GR', 'GAMMA', 'GR_EDIT', 'CGR', 'NGT', 'SGR', 'ECGR', 'GRC', 'GR.GAPI', 'GAPI', 'GR_GAPI', 'GAMWA'],
    # Resistividad Profunda (Inducción/Laterolog)
    'ILD': ['ILD', 'RES_DEEP', 'RT', 'LLD', 'RDEEP', 'RESISTIVITY', 'AT90', 'AF90', 'HDRS', 'ILD.OHM.M', 'RD', 'RES', 'RES_D', 'HRLD', 'RLA5'],
    # Resistividad Media/Somera (Opcional, mapeada si se requiere)
    'ILS': ['ILS', 'SN', 'RS', 'RES_MED', 'RES_SHALLOW', 'AT30', 'AF30', 'RINT', 'SN.OHM.M', 'SFL', 'SFLA', 'IMPH'],
    # Porosidad Neutrón
    'NPHI': ['NPHI', 'NEUT', 'TNPH', 'NPOR', 'CNPOR', 'NPHIS', 'NPHI.V/V', 'NPSS', 'NPHI_LS'],
    # Sónico (Delta T)
    'DT': ['DT', 'DTC', 'DTCO', 'DT4P', 'AC', 'SONIC', 'DT24', 'DTC_CO', 'DT.US/F', 'DT.US/FT', 'DTT', 'DTCOMP'],
    # Densidad Bulk
    'RHOB': ['RHOB', 'DEN', 'RHOZ', 'ZDEN', 'DENS', 'BDEN', 'RHOB.GM/C', 'RHOB.G/CC', 'DEN_B'],
    # Potencial Espontáneo
    'SP': ['SP', 'SPC', 'SP_EDC', 'SPBL', 'SPF', 'SP.MV', 'SS'],
    # Caliper
    'CALI': ['CALI', 'CAL', 'C13', 'DCAL', 'HCAL', 'CALIPER', 'CALI.IN', 'CAL.IN', 'CALS'],
    # Volumen de Arcilla
    'VSH': ['VSH', 'VSH_GR', 'VSHALE', 'VSH_DEC', 'VSH.DEC', 'VCL', 'VCLAY'],
    # Porosidad Efectiva
    'PHIE': ['PHIE', 'PHI_E', 'POR_EFF', 'PHIE_D', 'EPOR', 'PHIE.V/V', 'PIGE'],
    # Saturación de Agua
    'SW': ['SW', 'SW_ARCHIE', 'SATURATION', 'SWE', 'SXZ'],
    # Factor Fotoeléctrico
    'PEF': ['PEF', 'PE', 'PEFZ', 'PE.B/E'],
    # Permeabilidad
    'PERM': ['PERM', 'PERM_TIXIER', 'PERM2', 'KLOGH', 'KINT'],
    # Resistividad Zona Lavada (Flushed)
    'RXO': ['RXO', 'RX0', 'MSFL', 'MRLC1'] 
}

# Significado Físico de las Curvas (Geoscientific Logic)
CURVE_MEANING = {
    'GR': "Litología (Arcillosidad)",
    'ILD': "Fluidos (Saturación de Agua)",
    'NPHI': "Porosidad Total (Neutrón)",
    'DT': "Porosidad y Propiedades Mecánicas",
    'RHOB': "Porosidad Total (Densidad)",
    'SP': "Permeabilidad Relativa / Litología",
    'CALI': "Calidad de Hueco",
    'VSH': "Volumen de Arcilla",
    'PEF': "Litología (Mineralogía)",
    'PHIE': "Porosidad Efectiva",
    'SW': "Saturación de Agua"
}

# Límites Físicos Razonables (Geoscientific Logic)
# Se usan para filtrar outliers obvios o errores de medición
PHYSICAL_LIMITS = {
    'GR': (0, 500),       # GAPI
    'ILD': (0.1, 20000),  # Ohm.m (escala log)
    'NPHI': (-0.05, 0.6), # v/v (se permite leve negativo por efecto matriz)
    'DT': (30, 250),      # us/ft
    'RHOB': (1.0, 3.5),   # g/cc
    'CALI': (4, 30),      # pulgadas
    'SP': (-200, 200),    # mV
    'PEF': (0, 20)        # b/e
}

# Tolerancia de error aceptable para decisiones de "Reemplazo de Herramienta" (VOI v2.0)
# Si RMSE < Tolerancia, el dato sintético es operacionalmente indistinguible del real.
TOLERANCE_DICT = {
    'GR': 15.0,      # GAPI
    'ILD': 1.0,      # Ohm.m (Aprox lineal en rangos medios)
    'NPHI': 0.045,   # v/v (4.5 p.u.)
    'RHOB': 0.08,    # g/cc
    'DT': 10.0,      # us/ft
    'SP': 10.0,      # mV
    'CALI': 0.5,     # in
    'PEF': 0.5,      # b/e
    'VSH': 0.10,     # decimal
    'PHIE': 0.04,    # v/v
    'SW': 0.10,      # decimal
    'PERM': 10.0     # mD
}

unit_factors = {
    'M': 3.28084,  # Metros -> Pies
    'FT': 1.0
}

def get_std_name(curve_name):
    """Devuelve el nombre estándar para una curva dada (ej. 'GAMMA' -> 'GR')."""
    upper_name = curve_name.upper()
    # Búsqueda directa
    if upper_name in ALIAS_DICT:
        return upper_name
    # Búsqueda reversa en alias
    for std, aliases in ALIAS_DICT.items():
        if upper_name == std or upper_name in aliases:
            return std
    return None

def apply_physical_limits(df, curve, limits):
    """Aplica filtros de rango físico: reemplaza outliers por NaN."""
    min_val, max_val = limits
    mask = (df[curve] >= min_val) & (df[curve] <= max_val)
    # Logging de cuántos datos se pierden
    invalid_count = (~mask & df[curve].notna()).sum()
    if invalid_count > 0:
        pass # logger.debug(f"Filtrados {invalid_count} valores fuera de rango físico en {curve}")
    
    # Retorna la serie filtrada
    return df[curve].where(mask) 

def convert_depth_units(df, las_object):
    """
    Convierte la profundidad de Metros a Pies si el archivo LAS está en Metros.
    Estandariza internamente a Pies para consistencia con curvas como DT (us/ft).
    """
    depth_unit = None
    # Intentar leer la unidad de la curva de profundidad
    if las_object.curves and len(las_object.curves) > 0:
        depth_unit = las_object.curves[0].unit.upper().strip() if las_object.curves[0].unit else None
    
    # Fallback: leer del header STRT
    if not depth_unit:
        try:
            depth_unit = las_object.well.STRT.unit.upper().strip()
        except Exception:
            pass
    
    if not depth_unit:
        return df, 'UNKNOWN'
    
    # Detectar si está en metros
    meter_keywords = ['M', 'METER', 'METERS', 'METRE', 'METRES', 'MTS']
    if depth_unit in meter_keywords:
        # Buscar columna de profundidad
        depth_col = None
        for col in df.columns:
            if col.upper() in ['DEPT', 'DEPTH', 'MD', 'TVD', 'TDEP']:
                depth_col = col
                break
        if depth_col is None and len(df.columns) > 0:
            depth_col = df.columns[0]  # Asumir primera columna
        
        if depth_col:
            df[depth_col] = df[depth_col] * 3.28084
            logger.info(f"QC_UNITS: Convertida profundidad {depth_col} de M -> FT ({depth_unit})")
            return df, 'FT_CONVERTED'
    
    return df, depth_unit

def detect_casing(df, window=20, std_threshold=0.05):
    """
    Detecta tramos de casing (tubería) donde CALI es constante.
    En zonas entubadas, CALI ≈ diámetro del bit (constante), indicando que las 
    herramientas miden acero, no formación. Se marcan como NaN para excluir del entrenamiento.
    
    Criterio: desviación estándar de CALI en ventana móvil < threshold → casing.
    """
    if 'CALI' not in df.columns:
        return df, 0
    
    cali = df['CALI']
    if cali.isna().all():
        return df, 0
    
    # Calcular desviación estándar en ventana móvil
    rolling_std = cali.rolling(window=window, center=True, min_periods=5).std()
    
    # Zonas con CALI constante (cased hole)
    casing_mask = (rolling_std < std_threshold) & cali.notna()
    n_casing = casing_mask.sum()
    
    if n_casing > 0:
        # Curvas petrofísicas a invalidar en zonas de casing
        petro_curves = [c for c in df.columns if c in ALIAS_DICT or c in 
                        ['GR','ILD','ILS','NPHI','RHOB','DT','SP','PEF','VSH','PHIE','SW','PERM']]
        for curve in petro_curves:
            if curve in df.columns:
                df.loc[casing_mask, curve] = np.nan
        
        logger.info(f"QC_CASING: Detectados {n_casing} muestras en tramo de casing. Curvas invalidadas: {petro_curves}")
    
    return df, n_casing

def standardize_dataframe(df):
    """
    Tubería principal de limpieza y validación petrofísica.
    1. Estandariza Mnemónicos.
    2. Convierte Unidades (TODO: Si tuviéramos metadata de unidad).
    3. Aplica Límites Físicos.
    4. Elimina nulos de sistema (-999.25).
    """
    # 0. Limpieza inicial de nulos de sistema
    df = df.replace([-999.25, -9999, -999], np.nan)
    
    # Eliminación de duplicados exactos en nombres de columnas (mantiene la primera aparición)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 1. Renombrar Columnas
    rename_map = {}
    for col in df.columns:
        std = get_std_name(col)
        if std and std not in df.columns: # Evitar colisiones si ya existe
             rename_map[col] = std
    
    if rename_map:
        df = df.rename(columns=rename_map)
        # Eliminación de duplicados generados por el renombrado (ej. GR_EDIT -> GR y GAMMA -> GR)
        df = df.loc[:, ~df.columns.duplicated()]
        
    # 2. Validar Límites Físicos (Geoscientific Logic)
    # FORCE LOGGING CONFIG TO FILE (To be 100% sure we see output)
    try:
        import logging
        fh = logging.FileHandler("/app/output/debug.log")
        fh.setLevel(logging.ERROR)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
    except: pass

    for col in df.columns:
        # LOGGING INCONDICIONAL
        try:
            sample = df[col].iloc[0] if len(df) > 0 else 'EMPTY'
            logger.error(f"DEBUG_CLEAN_FORCE: Processing col '{col}' | dtype={df[col].dtype} | Sample: {sample}")
        except Exception as e:
            logger.error(f"DEBUG_CLEAN_FORCE: Error logging {col}: {e}")

        # LIMPIEZA TOTAL (SCORCHED EARTH)
        try:
            # 1. Force string conversion
            s_col = df[col].astype(str)
            # 2. Strip brackets and quotes and newlines using regex
            s_col = s_col.str.replace(r'[\[\]\'\" \n\r]', '', regex=True)
            # 3. Coerce to numeric
            df[col] = pd.to_numeric(s_col, errors='coerce')
            
            # Log result
            new_sample = df[col].iloc[0] if len(df) > 0 else 'EMPTY'
            logger.error(f"DEBUG_CLEAN_FORCE: Done col '{col}' | new_dtype={df[col].dtype} | NewSample: {new_sample}")
        except Exception as e:
            logger.error(f"DEBUG_CLEAN_FORCE: CRITICAL FAIL cleaning {col}: {e}")

        if col in PHYSICAL_LIMITS and col in df.columns:
            df[col] = apply_physical_limits(df, col, PHYSICAL_LIMITS[col])
            
    return df
