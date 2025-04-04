import joblib
from pathlib import Path
from typing import Optional


class SalaryModel:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self, model_path: Optional[str] = None):
        """Load trained model piplines"""
        default_path = Path(__file__).parent.parent / "salary_model.pkl"
        print(default_path)
        model_path = model_path or default_path
        self.model = joblib.load(model_path)

    def predict(self, data: dict) -> float:
        """Predict salary based on input data"""
        if self.model is None:
            raise ValueError(
                "Model not loaded. Please load the model before prediction."
            )

        # Convert input data to DataFrame
        import pandas as pd

        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = self.model.predict(input_df)
        # print(float(prediction[0]))

        return float(prediction[0])
