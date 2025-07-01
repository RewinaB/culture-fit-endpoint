from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(
    title="Culture Fit Predictor",
    description="API for predicting culture fit based on personality traits and job preferences using a trained Random Forest model.",
    version="0.0.1"
)

model = joblib.load("app/random_forest_model.joblib")

class PredictRequest(BaseModel):
    years_experience: float = 0
    agreeableness: float = 0
    conscientiousness: float = 0
    openness: float = 0
    extraversion: float = 0
    neuroticism: float = 0
    collaboration_score: float = 0
    innovation_preference: float = 0
    remote_preference: float = 0
    job_type_preference: str = "non-technical"



@app.get("/")
def read_root():
    return {"message": "Predicting culture fit"}

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict culture fit based on personality traits and preferences.

    - **years_experience**: Years of work experience (float)
    - **agreeableness**: Agreeableness trait score (float)
    - **conscientiousness**: Conscientiousness trait score (float)
    - **openness**: Openness trait score (float)
    - **extraversion**: Extraversion trait score (float)
    - **neuroticism**: Neuroticism trait score (float)
    - **collaboration_score**: Collaboration score (float)
    - **innovation_preference**: Innovation preference score (float)
    - **remote_preference**: Remote work preference score (float)
    - **job_type_preference**: Job type preference, "technical" or "non-technical" (string)

    **Returns:**  
    - `prediction`: 1 if culture fit, 0 if not  
    - `result`: "Culture Fit" or "Not Culture Fit"

    Example curl:
    ```
    curl -X 'POST' 'http://0.0.0.0:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
            "years_experience": 4.3,
            "agreeableness": 4.5,
            "conscientiousness": 4.0,
            "openness": 4.0,
            "extraversion": 3.3,
            "neuroticism": 1.8,
            "collaboration_score": 3.8,
            "innovation_preference": 3,
            "remote_preference": 0,
            "job_type_preference": "non-technical"
          }'
    ```
    """
    job_type_map = {"technical": 1, "non-technical": 0}
    job_type_encoded = job_type_map.get(request.job_type_preference, 0)

    features = [
        request.years_experience,
        request.agreeableness,
        request.conscientiousness,
        request.openness,
        request.extraversion,
        request.neuroticism,
        request.collaboration_score,
        request.innovation_preference,
        request.remote_preference,
        job_type_encoded,
    ]
    X = np.array(features).reshape(1, -1)
    prediction = int(model.predict(X))

    result = "Culture Fit" if prediction == 1 else "Not Culture Fit"
    return {"prediction": prediction, "result": result}
