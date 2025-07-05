# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """
    Loads the Boston Housing dataset as a Pandas DataFrame.
    This version uses explicit column indexing to ensure correct feature and target extraction,
    addressing potential parsing ambiguities with the raw data.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # The raw data has a unique structure:
    # Even-indexed rows (0, 2, ...) contain the first 9 features.
    # Odd-indexed rows (1, 3, ...) contain the next 4 features and the target.

    # Extract the first 9 features from the even rows.
    # We explicitly take columns 0 through 8 (9 columns).
    features_part1 = raw_df.values[::2, 0:9]

    # Extract the next 3 features from the odd rows.
    # These are typically columns 0, 1, 2 from the odd-indexed rows.
    features_part2 = raw_df.values[1::2, 0:3] # Corresponds to TAX, PTRATIO, B

    # The last feature (LSTAT) is typically column 3 from the odd-indexed rows.
    # We slice it as [:, 3:4] to ensure it remains a 2D array for hstack.
    features_part3 = raw_df.values[1::2, 3:4] # Corresponds to LSTAT

    # The target (MEDV) is typically column 4 from the odd-indexed rows.
    target = raw_df.values[1::2, 4]

    # Stack all feature parts horizontally: 9 + 3 + 1 = 13 columns
    data = np.hstack([features_part1, features_part2, features_part3])

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.
    """
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using MSE and R2 score.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2
