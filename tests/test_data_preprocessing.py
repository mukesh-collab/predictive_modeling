import pandas as pd
import pytest
from src.data_preprocessing import preprocess_data

import pandas as pd

def preprocess_data(data):
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.read_csv(data)

    # Continue preprocessing
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)

    return df

def test_preprocess_data_valid():
    # Valid DataFrame with missing values
    data = pd.DataFrame({
        'Age': [22, None, 35, None, 58],
        'Embarked': ['S', None, 'Q', 'C', None]
    })

    result = preprocess_data(data)
    assert result['Age'].isnull().sum() == 0
    assert result['Embarked'].isnull().sum() == 0
