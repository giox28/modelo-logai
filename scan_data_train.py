import os
import lasio
import pandas as pd
import json

DATA_DIR = "c:/temp/geoportal/proyecto_logai/data_train"

REQUIRED_INPUTS = {'GR', 'NPHI', 'ILD', 'RES', 'LLD', 'RT'} # At least one resistivity
TARGETS = {'DT', 'RHOB', 'NPHI'} # Potential targets

def scan_las_files():
    report = {}
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} does not exist.")
        return

    basins = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for basin in basins:
        basin_path = os.path.join(DATA_DIR, basin)
        files = [f for f in os.listdir(basin_path) if f.lower().endswith('.las')]
        
        basin_report = {
            "total_files": len(files),
            "key_wells": [],
            "partial_wells": [],
            "missing_essential": [],
            "all_curves": set()
        }
        
        print(f"Scanning basin: {basin} ({len(files)} files)")
        
        for f in files:
            try:
                las = lasio.read(os.path.join(basin_path, f))
                curves = set([c.mnemonic.upper() for c in las.curves])
                basin_report["all_curves"].update(curves)
                
                # Check inputs
                has_gr = 'GR' in curves or 'GAMMA' in curves
                has_res = any(r in curves for r in ['ILD', 'RES', 'LLD', 'RT', 'RDEEP'])
                has_nphi = 'NPHI' in curves or 'PHYN' in curves
                
                # Check targets
                has_dt = 'DT' in curves or 'DTCO' in curves
                has_rhob = 'RHOB' in curves or 'RHOZ' in curves
                
                well_info = {
                    "filename": f,
                    "curves": list(curves)
                }
                
                # Definition of Key Well for Training (needs Inputs + at least one Target)
                if has_gr and has_res and (has_dt or has_rhob):
                    basin_report["key_wells"].append(well_info)
                elif has_dt or has_rhob:
                     basin_report["partial_wells"].append(well_info)
                else:
                    basin_report["missing_essential"].append(well_info)
                    
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        basin_report["all_curves"] = list(basin_report["all_curves"])
        report[basin] = basin_report

    with open("scan_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Scan complete. Saved to scan_report.json")

if __name__ == "__main__":
    scan_las_files()
