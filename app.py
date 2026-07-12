from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib
import uvicorn
import os

app = FastAPI(title="AutoML++ Inference API", version="1.0")

preprocessor = None
model = None

@app.on_event("startup")
def load_artifacts():
    global preprocessor, model
    if not os.path.exists("models/preprocessor.pkl") or not os.path.exists("models/best_model.pkl"):
        raise RuntimeError("Artifacts missing! Run run_test.py first.")
    
    preprocessor = joblib.load("models/preprocessor.pkl")
    model = joblib.load("models/best_model.pkl")
    print("[INFO] Preprocessor and Model loaded into API memory.")

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def health_check():
    return {"status": "active", "message": "AutoML++ API is running."}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df = pd.DataFrame(request.data)
        X = preprocessor.transform(df)
        preds = model.predict(X)
        
        # Extract probabilities if it's a classification model
        probs = None
        if hasattr(model, "predict_proba"):
            # .tolist() converts numpy arrays to standard Python lists for JSON serialization
            probs = model.predict_proba(X).tolist()

        return {
            "status": "success",
            "predictions": preds.tolist(),
            "probabilities": probs
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)