import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
import random
import mlflow
import os
from logisticRegression_3 import Ridge, Normal, RidgePenalty

# model = pk.load(open('st125051-car-prediction-a3.pkl', 'rb'))
from dotenv import load_dotenv
load_dotenv()

username = os.getenv('MLFLOW_TRACKING_USERNAME')
password = os.getenv('MLFLOW_TRACKING_PASSWORD')
model_uri = f'http://{username}:{password}@mlflow.ml.brain.cs.ait.ac.th/#/experiments/217142467269316447/runs/05ea8a55fef44f4d8b14cd6e78c4fc9d/artifacts/model/st125051-a3-model.pkl'

# Load the model from MLflow
try:
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"Failed to load the model from MLflow: {str(e)}")

# model_name = "st125051-a3-model"
# model_version = 1
# model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")  

st.header('ML Car Prediction A3 Assignment')

st.subheader('Car Price Prediction')

default_values = {'engine': 1248, 'mileage': 23.4 , 'year': 2014}
num_cols = ['engine', 'mileage', 'year']

def get_X(engine, mileage, year):
    features={
        'engine': engine,
        'mileage': mileage,
        'year': year
    }

    for feature in features:
        if not features[feature]:
            features[feature]=default_values[feature]
        elif feature in num_cols:
            if features[feature]<0:
                features[feature] = default_values[feature]
    X = pd.DataFrame(features, index=0)

    print(X)
    return X.to_numpy, features

def predict_car_price(year, mileage, engine):
    input_data = pd.DataFrame(
        [[engine, mileage, year]],
        columns=['engine', 'mileage', 'year']
    )
    
    car_price = model.predict(input_data)
    
    return car_price[0]

year = st.number_input('Manufacturing Year', step=1)
mileage = st.number_input('Car Mileage')
engine = st.number_input('Engine Capacity')

if st.button("Predict Car Price"):
    predicted_price = predict_car_price(year, mileage, engine)
    st.markdown(f'Estimated Car Price is : {predicted_price}')
