from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import JSONResponse


import joblib
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")


model = joblib.load("../salary_model.pkl")
category_mappings = joblib.load("../category_mappings.pkl")
# one_hot_columns = joblib.load("../one_hot_encoded.pkl")
# label_encoder = joblib.load("../label_encoder.pkl")


class SalaryRequest(BaseModel):
    years_of_experience: float
    education_level: int
    job_title: float
    age: int
    gender: int


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_salary")
async def predict_salary(data: SalaryRequest):
    try:
        features = {
            "Age": data.age,
            "Gender": data.gender,
            "Education Level": data.education_level,
            "Job Title": data.job_title,
            "Years of Experience": data.years_of_experience,
        }
        import pandas as pd

        input_df = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(input_df)

        # Return the prediction
        return {"predicted_salary": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# @app.post("/predict_salary")
# async def predict_salary(request: SalaryRequest):
#     try:
#         input_data = np.array(
#             [
#                 [
#                     request.years_of_experience,
#                     request.age,
#                     request.education_level,
#                     request.job_title,
#                     request.gender,
#                 ]
#             ]
#         )  # Adjust based on your features
#         prediction = model.predict(input_data)

#         return {"predicted_salary": prediction[0]}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
