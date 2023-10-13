import joblib
import pandas as pd

# Load the trained model
model = joblib.load('trained_model.pkl')

# Collect user input
income = float(input("Enter Avg. Area Income: "))
house_age = float(input("Enter Avg. Area House Age: "))
num_rooms = float(input("Enter Avg. Area Number of Rooms: "))
num_bedrooms = float(input("Enter Avg. Area Number of Bedrooms: "))
population = float(input("Enter Area Population: "))

# Create a DataFrame with user input
user_input_data = pd.DataFrame({
    'Avg. Area Income': [income],
    'Avg. Area House Age': [house_age],
    'Avg. Area Number of Rooms': [num_rooms],
    'Avg. Area Number of Bedrooms': [num_bedrooms],
    'Area Population': [population]
})

# Use the trained model to make predictions
predicted_price = model.predict(user_input_data)

# Display the predicted house price
print(f"Predicted House Price: ${predicted_price[0]:,.2f}")
