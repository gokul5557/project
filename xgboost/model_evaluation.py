import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import os

def evaluate_model():
    # Get the current directory
    current_directory = os.path.dirname(__file__)

    # Define the file paths for the preprocessed data
    X_train_path = os.path.join(current_directory, 'X_train.csv')
    X_test_path = os.path.join(current_directory, 'X_test.csv')
    y_train_path = os.path.join(current_directory, 'y_train.csv')
    y_test_path = os.path.join(current_directory, 'y_test.csv')

    # Load the preprocessed data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    # Feature selection (example using SelectKBest)
    selector = SelectKBest(score_func=f_regression, k='all')  # Use 'all' to keep all features
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Fit the model to your training data with selected features
    model = joblib.load(os.path.join(current_directory, 'trained_model.pkl'))
    model.fit(X_train_selected, y_train)

    # Make predictions on the test data with selected features
    y_pred = model.predict(X_test_selected)

    # Calculate Mean Absolute Error (MAE) for evaluation
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")

# Call the evaluate_model function to evaluate the model
if __name__ == "__main__":
    evaluate_model()
