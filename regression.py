# regression.py
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
# If you are working on the hyper_branch, remember to also import GridSearchCV
# from sklearn.model_selection import GridSearchCV

from utils import load_data, split_data, evaluate_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_evaluate_regression_models():
    logging.info("Loading data...")
    df = load_data()
    logging.info("Data loaded successfully.")

    # --- START: Added Data Cleaning Steps ---
    logging.info("Checking for missing values and non-numeric data...")
    # Drop rows with any NaN values. This is a simple handling strategy.
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    rows_after_dropna = df.shape[0]
    if initial_rows > rows_after_dropna:
        logging.warning(f"Dropped {initial_rows - rows_after_dropna} rows due to missing values.")

    # Convert all columns to numeric, coercing errors to NaN and then dropping them
    # This helps catch any non-numeric data that might have slipped through
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True) # Drop any new NaNs created by coercion

    if df.empty:
        logging.error("DataFrame is empty after cleaning. Cannot proceed with training.")
        return # Exit if no data is left

    logging.info(f"Data shape after cleaning: {df.shape}")
    # --- END: Added Data Cleaning Steps ---


    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    logging.info("Data split complete.")

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42), # Added random_state for reproducibility
        "Random Forest Regressor": RandomForestRegressor(random_state=42) # Added random_state for reproducibility
        # Add more models here, e.g., "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        logging.info(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            logging.info(f"{name} trained.")

            mse, r2 = evaluate_model(model, X_test, y_test)
            results[name] = {"MSE": mse, "R2": r2}
            logging.info(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
        except Exception as e:
            logging.error(f"Error training {name}: {e}")
            results[name] = {"Error": str(e)}


    logging.info("\n--- Regression Model Performance Comparison (No Hyperparameter Tuning) ---")
    for name, metrics in results.items():
        logging.info(f"{name}:")
        if "Error" in metrics:
            logging.info(f"  Error: {metrics['Error']}")
        else:
            logging.info(f"  MSE: {metrics['MSE']:.4f}")
            logging.info(f"  R2: {metrics['R2']:.4f}")
    logging.info("-------------------------------------------------------------------")

    # You might want to save these results to a file for the report
    with open("regression_results.txt", "w") as f:
        f.write("--- Regression Model Performance Comparison (No Hyperparameter Tuning) ---\n")
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            if "Error" in metrics:
                f.write(f"  Error: {metrics['Error']}\n")
            else:
                f.write(f"  MSE: {metrics['MSE']:.4f}\n")
                f.write(f"  R2: {metrics['R2']:.4f}\n")
        f.write("-------------------------------------------------------------------\n")

if __name__ == "__main__":
    train_and_evaluate_regression_models()
