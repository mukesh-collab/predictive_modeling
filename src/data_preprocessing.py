"""
Module for data loading and preprocessing.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(file_path):
    """
    Loads and preprocesses the dataset.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: Processed training and testing data (X_train, X_test, y_train, y_test).
    """
    # Load dataset
    data = pd.read_csv(file_path)

    # Drop unnecessary columns
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # Handle missing values
    data.loc[:, 'Age'] = data['Age'].fillna(data['Age'].median())
    data.loc[:, 'Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    # Features and target
    X = data.drop(['Survived'], axis=1)
    y = data['Survived']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
