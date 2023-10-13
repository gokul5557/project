import pandas as pd
from xgboost import XGBRegressor  # Import XGBoost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from data_preprocessing import load_and_preprocess_data
import os

def train_model():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('USA_Housing.csv')

    # Create an XGBoost Regressor model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)

    # Fit the model to your training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate Mean Absolute Error (MAE) for evaluation
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

    # Save the trained model
    import joblib
    current_directory = os.path.dirname(__file__)  # Get the current directory
    joblib.dump(model, os.path.join(current_directory, 'trained_model.pkl'))

# Call the train_model function to train the XGBoost model
if __name__ == "__main__":
    train_model()
