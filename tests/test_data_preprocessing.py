import pandas as pd
import pytest
from src.data_preprocessing import preprocess_data

def test_preprocess_data_valid():
    # Valid DataFrame with missing values
    data = pd.DataFrame({
        'Age': [22, None, 35, None, 58],
        'Embarked': ['S', None, 'Q', 'C', None]
    })

    result = preprocess_data(data)

    # Check missing values are filled
    assert result['Age'].isnull().sum() == 0, "Age column still has missing values."
    assert result['Embarked'].isnull().sum() == 0, "Embarked column still has missing values."

    # Check encoding
    assert result['Embarked'].dtype.name == 'int8', "Embarked column was not encoded properly."

def test_preprocess_data_empty():
    # Empty DataFrame
    data = pd.DataFrame(columns=['Age', 'Embarked'])

    result = preprocess_data(data)

    # Check the result remains empty
    assert result.empty, "Preprocessed data should be empty for empty input."

def test_preprocess_data_invalid():
    # Invalid DataFrame (missing required columns)
    data = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Ticket': [1, 2, 3]
    })

    with pytest.raises(KeyError):
        preprocess_data(data)
