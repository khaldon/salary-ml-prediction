from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from model import SalaryModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")


model = SalaryModel()


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
        # Make prediction
        prediction = model.predict(features)
        rounded = round(prediction)
        return f"predicted salary: ${rounded:,}"

        # Return the prediction
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
