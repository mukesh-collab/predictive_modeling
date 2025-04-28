"""
Module for data loading and preprocessing with logging.
"""
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configure logging
logging.basicConfig(filename='preprocessing.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def preprocess_data(file_path):
    """
    Loads and preprocesses the dataset.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: Processed training and testing data (X_train, X_test, y_train, y_test).
    """
    try:
        logger.info("Starting data preprocessing.")

        # Load dataset
        logger.info(f"Loading dataset from {file_path}.")
        data = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully.")

        # Drop unnecessary columns
        columns_to_drop = ['Name', 'Ticket', 'Cabin']
        logger.info(f"Dropping unnecessary columns: {columns_to_drop}.")
        data = data.drop(columns_to_drop, axis=1)
        logger.info("Unnecessary columns dropped.")

        # Handle missing values
        logger.info("Filling missing values for 'Age' with the median.")
        data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median())
        logger.info("Filling missing values for 'Embarked' with the mode.")
        data.loc[:, 'Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
        logger.info("Missing values filled.")

        # Encode categorical variables
        logger.info("Encoding categorical variables: 'Sex' and 'Embarked'.")
        label_encoder = LabelEncoder()
        data['Sex'] = label_encoder.fit_transform(data['Sex'])
        data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
        logger.info("Categorical variables encoded.")

        # Features and target
        logger.info("Separating features and target variable.")
        X = data.drop(['Survived'], axis=1)
        y = data['Survived']

        # Split the dataset
        logger.info("Splitting the dataset into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Dataset split into training and testing sets.")

        # Scale numerical features
        logger.info("Scaling numerical features.")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Numerical features scaled.")

        logger.info("Data preprocessing completed successfully.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        raise
