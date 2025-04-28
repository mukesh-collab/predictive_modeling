"""
Module for model training.
"""
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    """
    Trains a Random Forest Classifier.

    Args:
        X_train (array): Training features.
        y_train (array): Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
