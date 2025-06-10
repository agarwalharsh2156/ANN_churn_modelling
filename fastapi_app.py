from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# --- Global mappings for categorical features ---
# These MUST exactly match the mappings used during your model training
GENDER_MAP = {'Female': 0, 'Male': 1}
GEOGRAPHY_MAP = {'France': 0, 'Spain': 1, 'Germany': 2}

# Load the trained model and scaler when the app starts
try:
    # UPDATED: Model filename
    model = joblib.load('churn_prediction_model.pkl')
    # NEW: Load the scaler
    scaler = joblib.load('scaler.pkl')
    print("FastAPI app: Model and Scaler loaded successfully!")
except Exception as e:
    print(f"FastAPI app: Error loading model or scaler: {e}")
    model = None
    scaler = None # Ensure scaler is also None if loading fails

# Define the expected input data structure for your API
# UPDATED: Gender and Geography are now strings
class ChurnPredictionInput(BaseModel):
    CreditScore: float
    Geography: str  # Now accepts 'France', 'Spain', 'Germany'
    Gender: str     # Now accepts 'Female', 'Male'
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    # Ensure these match the order and exact names of columns in your X_train

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: ChurnPredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded on server.")

    try:
        # NEW: Convert string inputs to numerical using predefined mappings
        # Check if input values are valid before mapping
        if data.Gender not in GENDER_MAP:
            raise ValueError(f"Invalid Gender value: '{data.Gender}'. Must be 'Female' or 'Male'.")
        mapped_gender = GENDER_MAP[data.Gender]

        if data.Geography not in GEOGRAPHY_MAP:
            raise ValueError(f"Invalid Geography value: '{data.Geography}'. Must be 'France', 'Spain', or 'Germany'.")
        mapped_geography = GEOGRAPHY_MAP[data.Geography]

        input_list = [
            data.CreditScore,
            mapped_geography,   # Use mapped value
            mapped_gender,      # Use mapped value
            data.Age,
            data.Tenure,
            data.Balance,
            data.NumOfProducts,
            data.HasCrCard,
            data.IsActiveMember,
            data.EstimatedSalary
        ]

        # Convert to numpy array and reshape for single prediction
        input_data = np.array(input_list).reshape(1, -1)

        # NEW: Scale the input data using the loaded StandardScaler
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the scaled data``
        prediction_proba = model.predict(input_data_scaled)[0][0] # Get the single probability value
        predicted_class = (prediction_proba > 0.5).astype(int)
        result = ""
        if(predicted_class==0):
            result="No Churn"
        else:
            result="Churn"


        return {
            "prediction": result,
            "probability": float(prediction_proba)
        }

    except ValueError as ve: # Catch specific value errors from mapping
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        # General error for unexpected issues during prediction
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {e}")

# Basic route for testing if the API is alive
@app.get("/")
async def read_root():
    return {"message": "FastAPI ML API is running!"}