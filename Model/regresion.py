import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from Model.DTO.ClimateDTO import ClimateDTO
import os
class model_reg:

    def __init__(self) -> None:
        self.__linear_reg: LinearRegression= LinearRegression()
        return
    
    def clean_dataset(self):
        current_directory = os.getcwd()
        definitivo= pd.read_csv(current_directory+"/Model/weather.csv",sep=",")
        df_dummies= pd.get_dummies(definitivo, columns=['Description'])
        return df_dummies
    
    def create_model(self)->None:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("regression-climate-information")
        definitivo= self.clean_dataset()

        X=definitivo[["Description_Warm","Rain","Visibility_km","Wind_Speed_kmh","Wind_Bearing_degrees", "Description_Cold","Humidity","Description_Normal"]]
        y=definitivo.Temperature_c


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=34)
        self.__linear_reg.fit(X_train,y=y_train)
        ajustados=self.__linear_reg.predict(X_train)
        y_pred=self.__linear_reg.predict(X_test)
        signature = infer_signature(X_train, y_pred)

        mlflow.log_metric("mse- train", mean_squared_error(y_train, ajustados))
        mlflow.log_metric("mse- test", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2-train ", r2_score(y_train, ajustados))
        mlflow.log_metric("r2-test", r2_score(y_test, y_pred))

        mlflow.sklearn.log_model(self.__linear_reg, "model", signature=signature)

    def predict_data(self,data: ClimateDTO):
        serie= pd.DataFrame(data.__dict__, index=[0])
        return self.__linear_reg.predict(serie)[0]
