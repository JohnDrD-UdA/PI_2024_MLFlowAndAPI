from Model.DTO.ClimateDTO import ClimateDTO
from Model.regresion import model_reg
from fastapi import FastAPI	

lr=model_reg()
lr.create_model()
app= FastAPI()

@app.post("/predict")
async def predict(data:ClimateDTO):
    return lr.predict_data(data)