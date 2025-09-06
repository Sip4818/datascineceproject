import joblib
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        model_path = Path("artifacts/model_trainer/model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)  # Use load, not dump

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction
