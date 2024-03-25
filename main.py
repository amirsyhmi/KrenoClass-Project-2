from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import time

class LoanApplication(BaseModel):
    Loan_ID: Union[str,None]
    Gender: Union[str,None]
    Married: Union[str,None]
    Dependents: Union[str,None]
    Education: Union[str,None]
    Self_Employed: Union[str,None]
    ApplicantIncome: Union[float,None]
    CoapplicantIncome: Union[float,None]
    LoanAmount: Union[float,None]
    Loan_Amount_Term: Union[float,None]
    Credit_History: Union[float,None]
    Property_Area: Union[str,None]

class Prediction(BaseModel):
    label: Union[int, str]
    value: Union[str, None]

class APIResponse(BaseModel):
    status: str
    error_code: int
    prediction: Prediction
    time_taken: str

app = FastAPI()



@app.post("/predict", response_model=APIResponse)
async def predict(request: LoanApplication):
    data = request.model_dump()
    df = pd.DataFrame([data])

    # fill empty values 
    df['Credit_History'].fillna(1, inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['Dependents'].fillna("0", inplace=True)
    df['Gender'].fillna("Male", inplace=True)

    # remove anyu data is still null value
    df = df.dropna()

    if df.empty:
        return {"error":"Data is not valid format"}
    
    # data preprocessing and mapping
    df["Gender"] = df['Gender'].apply(lambda x:1 if x == 'Male' else 0)
    df["Married"] = df['Married'].apply(lambda x:1 if x == 'Married' else 0)
    df["Dependents"] = df['Dependents'].map({'0': 0, '1':1, '2':2, '3+': 3})
    df["Education"] = df['Education'].apply(lambda x: 1 if x == 'Graduate' else 0)
    df['Self_Employed'] = df['Self_Employed'].apply(lambda x:1 if x == 'No' else 0)
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Rural':0, 'Semiurban': 1})

    

    # load model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    start_time = time.time()
    input = df.drop(columns=['Loan_ID'])
    prediction = model.predict(input)
    prediction = prediction.tolist()
    end_time = time.time()  # End time after prediction
    time_taken = str(round((end_time - start_time)*1000))  # Time taken for prediction in miliseconds

    # Checf if prediction was successful
    if prediction is not None:
        status = "success"
        error_code = 0
        label = prediction[0]
        value = "Approve" if prediction[0] == 1 else "Reject"
    else:
        status = "failed"
        error_code = 1
        label = "N/A"
        value = "N/A"


    return APIResponse(
        status=status,
        error_code=error_code,
        prediction=Prediction(label=label, value=value),
        time_taken= f"{time_taken}ms"
    )
