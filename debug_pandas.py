import pandas as pd
try:
    s = pd.Series(["[5.18086E0]"])
    print(f"Original: {s}")
    cleaned = pd.to_numeric(s, errors='coerce')
    print(f"Cleaned: {cleaned}")
except Exception as e:
    print(f"Error: {e}")
