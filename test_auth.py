import requests

url = "http://localhost:8003/token"
data = {"username": "admin", "password": "Bien.2026*"}

try:
    response = requests.post(url, data=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
