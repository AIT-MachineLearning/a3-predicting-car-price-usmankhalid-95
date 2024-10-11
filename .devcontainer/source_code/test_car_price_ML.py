import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from logisticRegression_3 import Ridge, Normal, RidgePenalty
from UI_3 import predict_car_price, get_X
import mlflow


# Define labels
# poss

# def load_model():
#     with open('./st125051-car-prediction-a3.pkl', 'rb') as file:
#         print("file path:", file)
#         model = pickle.load(file)
#     return model

# sample_data = {
#     'engine': 1248.0,
#     'mileage': 23.5,
#     'year': 2014  
# }

# def main():
#     model = load_model()
    
#     # Create a DataFrame from sample data
#     sample_df = pd.DataFrame([sample_data])
    
#     # Get prediction from the model
#     prediction = model.predict(sample_df)
    
#     # Assuming the prediction is an index corresponding to the labels
#     predicted_label = labels[int(prediction[0])]  # Convert prediction to index and get label
#     print(f"Estimated Car Price category is: {predicted_label}")

# def test_load_model():
#     model = load_model()
#     assert model is not None  # Example test to check if the model loads

# def test_prediction():
#     model = load_model()
#     sample_df = pd.DataFrame([sample_data])
#     prediction = model.predict(sample_df)
#     assert prediction is not None  # Example test to check if prediction returns a value

# if __name__ == "__main__":
#     main()

feature_vals = [1248, 23.4, 2014]
labels = ['Cheap', 'Average', 'Expensive', 'Premium']
possible_outputs = [label for label in labels]

def test_get_X():
    X, features = get_X(*feature_vals)
    assert X.shape == (1, 4)
    assert X.dtype == np.float64


def test_selling_price():
    output = predict_car_price(1248, 23.4, 2014)
    #assert output[0] in possible_outputs
    if isinstance(output, str):
        output = np.array([output])
        
    assert output.shape == (1,), f'Expected output shape: (1,), Output shape received: {output.shape}'

# def main():
#     predicted_price = predict_car_price(1248, 23.4, 2014)
#     print(predicted_price)

# if __name__ == "__main__":
#     main()