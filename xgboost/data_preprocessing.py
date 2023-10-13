import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Step 1: Handling Missing Values
    # Check for missing values in each column
    missing_values = data.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Remove rows with missing values (if desired)
    data_cleaned = data.dropna()

    # Step 3: Cleaning the Dataset (Other cleaning operations)
    # Drop duplicate rows
    data_cleaned = data_cleaned.drop_duplicates()

    # Split the data into features (X) and target (y)
    X = data_cleaned[['Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
    y = data_cleaned['Price']

    # Data preprocessing steps
    # Standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data to files
    current_directory = os.path.dirname(__file__)  # Get the current directory
    pd.DataFrame(X_train).to_csv(os.path.join(current_directory, 'X_train.csv'), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(current_directory, 'X_test.csv'), index=False)
    pd.DataFrame(y_train).to_csv(os.path.join(current_directory, 'y_train.csv'), index=False)
    pd.DataFrame(y_test).to_csv(os.path.join(current_directory, 'y_test.csv'), index=False)

    # Visualization and Exploration
    # Visualization: Histogram of 'Avg. Area House Age'
   #  plt.figure(figsize=(10, 6))
    # sns.histplot(X_train[:, 0], bins=30, kde=True)  # Index 0 corresponds to 'Avg. Area House Age'
    # plt.title('Histogram of Avg. Area House Age')
    # plt.xlabel('Avg. Area House Age')
    # plt.ylabel('Frequency')
    # plt.show()

    # Visualization: Pairplot for selected columns
   #  data_for_pairplot = pd.DataFrame(X_train, columns=['Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population'])
   #  data_for_pairplot['Price'] = y_train
   #  sns.pairplot(data_for_pairplot)
    # plt.show()

    # Visualization: Correlation heatmap
   #  correlation_matrix = data_for_pairplot.corr()
  #   plt.figure(figsize=(10, 6))
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
   #  plt.title('Correlation Heatmap')
   #  plt.show()

    return X_train, X_test, y_train, y_test
