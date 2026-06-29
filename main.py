from fastapi import FastAPI 
from pydantic import BaseModel
from typing import Literal
import joblib
app=FastAPI()
model=joblib.load('models/heart_model.joblib')

target={0:'No Heart Disease',1:'Heart Disease'}
class Heart(BaseModel):
    age:int
    sex:Literal["male","female"]
    cp:int
    trestbps:int
    chol:int
    fbs:int
    restecg:int
    thalach:int
    exang:int
    oldpeak:float
    slope:int
    ca:int
    thal:int

@app.post('/predict')
def prediction(x:Heart):
    sex=1 if x.sex== "male" else 0
    sample=[[x.age,sex,x.cp,x.trestbps,x.chol,x.fbs,x.restecg,x.thalach,x.exang,x.oldpeak,x.slope,x.ca,x.thal]]
    prediction=model.predict(sample)
    result=target[prediction[0]]
    return{
        "prediction":result
    }

    


