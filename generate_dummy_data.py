import os
import numpy as np
import pandas as pd
import lasio

def generate_synthetic_las(basin_name, num_wells=5, output_dir="data_train"):
    """
    Genera archivos LAS sintéticos con correlaciones simples para probar el entrenamiento.
    """
    basin_dir = os.path.join(output_dir, basin_name)
    os.makedirs(basin_dir, exist_ok=True)
    
    print(f"Generando {num_wells} pozos sintéticos en {basin_dir}...")
    
    for i in range(num_wells):
        # Crear estructura LAS
        las = lasio.LASFile()
        las.well.WELL = f"WELL-{basin_name}-{i+1}"
        las.well.UWI = f"{basin_name}-{i+1}" # Importante para GroupKFold
        
        # Profundidad (MD)
        depth = np.arange(1000, 2000, 0.5)
        n_samples = len(depth)
        
        # Generar curvas input simuladas
        # GR: Gamma Ray (0 - 150)
        # NPHI: Neutron Porosity (0.45 - 0.05) correlated with GR (shale has high porosity reading usually)
        # ILD: Resistivity (log scale 0.2 - 2000)
        
        # Simulamos facies con una onda
        facies_pattern = np.sin(depth * 0.05) 
        
        gr = 75 + 50 * facies_pattern + np.random.normal(0, 5, n_samples)
        gr = np.clip(gr, 0, 200)
        
        nphi = 0.25 + 0.15 * facies_pattern + np.random.normal(0, 0.02, n_samples)
        nphi = np.clip(nphi, 0.01, 0.5)
        
        # ILD anti-correlacionado con GR (arena limpia -> alta res si tiene HC, arcilla -> baja res)
        ild_log = 1 - 0.5 * facies_pattern + np.random.normal(0, 0.1, n_samples)
        ild = 10 ** ild_log
        
        # Generar Targets (DT, RHOB) basados en inputs (Física de rocas simple)
        # RHOB: Densidad. 2.65 (matrix) * (1-phi) + 1.0 (fluid) * phi
        # Simplificación: rhob inversamente prop. a NPHI
        rhob = 2.65 - 1.65 * nphi + np.random.normal(0, 0.05, n_samples)
        
        # DT: Sónico. Wyllie time average. DT_matrix (~50) + phi * (DT_fluid (~189) - DT_matrix)
        dt = 50 + nphi * (189 - 50) + np.random.normal(0, 2, n_samples)
        
        # Añadir curvas al LAS
        las.append_curve('DEPT', depth, unit='FT')
        las.append_curve('GR', gr, unit='API', descr='Gamma Ray')
        las.append_curve('NPHI', nphi, unit='V/V', descr='Neutron Porosity')
        las.append_curve('ILD', ild, unit='OHMM', descr='Deep Resistivity')
        
        # Targets
        las.append_curve('RHOB', rhob, unit='G/C3', descr='Density')
        las.append_curve('DT', dt, unit='US/FT', descr='Sonic Co')
        
        # Guardar
        filename = f"well_{i+1}.las"
        las.write(os.path.join(basin_dir, filename), version=2.0)

if __name__ == "__main__":
    generate_synthetic_las("Llanos", num_wells=6)
    generate_synthetic_las("VMM", num_wells=4)
