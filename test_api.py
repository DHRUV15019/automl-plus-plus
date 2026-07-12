import requests
import json

# Load the raw sample data we saved earlier
try:
    with open("models/sample_request.json", "r") as f:
        sample_data = json.load(f)
except FileNotFoundError:
    print("[ERROR] sample_request.json not found in models/ folder.")
    exit(1)

print("Sending this raw data to API:\n", json.dumps(sample_data, indent=2))

# Hit the local FastAPI endpoint
url = "http://127.0.0.1:8000/predict"
response = requests.post(url, json={"data": sample_data})

if response.status_code == 200:
    print("\n[SUCCESS] API Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"\n[ERROR] {response.status_code}: {response.text}")