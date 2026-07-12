import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime

from src.ingestion import DataIngestor
from src.preprocessing import AutoPreprocessor
from src.model_search import AutoModelSearch
from src.explainability import AutoExplainer

# CONSTANTS (Easier to swap datasets now)
DATASET_PATH = "data/titanic.csv"
TARGET_COL = "Survived"

# 1. Ingestion
print("\n--- INGESTION STAGE ---")
ingestor = DataIngestor(file_path=DATASET_PATH, target_col=TARGET_COL)
df = ingestor.load_data()
task = ingestor.detect_task_type()

# Split before preprocessing (Strict data leakage prevention)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET_COL] if "classification" in task else None)

# 2. Preprocessing
print("\n--- PREPROCESSING STAGE ---")
preprocessor = AutoPreprocessor(target_col=TARGET_COL)
X_train, y_train = preprocessor.fit_transform(train_df)
X_test, y_test = preprocessor.transform(test_df)

# 3 & 4. Model Search and Training
print("\n--- MODEL SEARCH & TRAINING STAGE ---")
searcher = AutoModelSearch(task_type=task, n_trials=20, timeout=120)
best_model = searcher.optimize_and_train(X_train, y_train)

# 5. Evaluation
print("\n--- FINAL EVALUATION ---")
test_metric = searcher.evaluate(X_test, y_test)

# 6. SHAP Explainability
print("\n--- EXPLAINABILITY STAGE ---")
explainer = AutoExplainer(model=best_model, X_train=X_train, task_type=task)
explainer.calculate_shap()
explainer.plot_summary()
explainer.plot_local_explanation(instance_index=0) # Yeh nayi line hai

# 7. Model Serialization & Metadata Logging
print("\n--- SERIALIZATION STAGE ---")
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(best_model, "models/best_model.pkl")

# Save metadata log
metadata = {
    "timestamp": datetime.now().isoformat(),
    "dataset": DATASET_PATH,
    "task_type": task,
    "best_params": searcher.best_params,
    "test_metric": float(test_metric)
}
with open("models/run_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Save a small sample row for API testing
test_df.drop(columns=[TARGET_COL]).head(1).to_json("models/sample_request.json", orient="records")
print("[INFO] Preprocessor, Model, Metadata, and sample request saved to 'models/' folder successfully.")