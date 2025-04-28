"""
Unit tests for data preprocessing module.
"""
import os
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    data_path = os.path.join("data", "titanic.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0
