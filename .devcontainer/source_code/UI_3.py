import pandas as pd 
import numpy as np 
import pickle as pk 
import streamlit as st
import random
from logisticRegression_3 import Ridge, Normal, RidgePenalty

model = pk.load(open('st125051-car-prediction-a3.pkl', 'rb'))

st.header('ML Car Prediction A3 Assignment')

st.subheader('Car Price Prediction')
year = st.number_input('Manufacturing Year', step=1)
mileage = st.number_input('Car Mileage')
engine = st.number_input('Engine Capacity')

labels = ['Cheap', 'Average', 'Expensive', 'Very Expensive']

if st.button("Predict Car Price"):
    input_data = pd.DataFrame(
        [[engine, mileage, year]],
        columns=['engine', 'mileage', 'year']
    )
    
    car_price = model.predict(input_data)
    
    price_label = labels[int(car_price[0])]
    
    st.markdown(f'Predicted Class: {int(car_price[0])}')
    st.markdown(f'Estimated Car Price Label is: {price_label}')
