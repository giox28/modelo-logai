import requests
import os
import json

url = "http://localhost:8003/process-well"
file_path = r"c:/temp/geoportal/proyecto_logai/data_train/cauca_patia/ANH CAUCA 10 STS.las"

payload = {
    "basin_name": "cauca_patia", # Lowercase directory name match?
    "project_type": "oil",
    "target_curves": "DT,RHOB"
}

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

print(f"Sending request to {url} with {file_path}...")
try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, data=payload, files=files)

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Keys:", data.keys())
        if "synthetic_data" in data:
            syn = data["synthetic_data"]
            print("Synthetic Data Keys:", syn.keys())
            for k, v in syn.items():
                print(f"Target {k} type: {type(v)}")
                if isinstance(v, dict):
                    print(f"  Keys inside {k}: {list(v.keys())}")
                    p50 = v.get("P50_SYN")
                    if p50 is not None:
                        print(f"  P50 length: {len(p50)}")
                        # Check for non-nulls
                        non_nulls = [x for x in p50 if x is not None]
                        print(f"  P50 non-null count: {len(non_nulls)}")
                        print(f"  P50 sample: {p50[:10]}")
                    else:
                        print("  P50 is None")
                else:
                    print(f"  Value type: {type(v)}")
        else:
            print("synthetic_data missing!")
        
        if "depth_data" in data:
            d = data["depth_data"]
            print(f"Depth Data length: {len(d)}")
            print(f"Depth sample: {d[:5]}")
            
    else:
        print("Error Response:", response.text)

except Exception as e:
    print(f"Exception: {e}")
