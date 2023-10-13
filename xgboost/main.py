import os
from data_preprocessing import load_and_preprocess_data
from model_training import train_model
from model_evaluation import evaluate_model

if __name__ == "__main__":
    # Step 1: Load and preprocess the data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('USA_Housing.csv')

    # Step 2: Train the XGBoost regression model
    train_model()

    # Step 3: Evaluate the model
    evaluate_model()
