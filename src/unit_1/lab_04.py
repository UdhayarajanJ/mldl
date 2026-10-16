# Demonstrate various data pre-processing techniques for an iris dataset.

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def lab_04():
    """
    Lab 04: Data Pre-processing Techniques

    This lab demonstrates various data pre-processing techniques using the Iris dataset.
    It covers loading the dataset, checking for missing values, feature scaling, and
    splitting the dataset into training and testing sets.

    Steps:
    1. Load the Iris dataset and convert it into a Pandas DataFrame.
    2. Check for missing values and handle them if necessary.
    3. Separate features (X) and target labels (Y).
    4. Apply feature scaling (standardization) to the features.
    5. Split the dataset into training and testing sets.
    """
    # ==========================================
    # 1. LOAD AND INSPECT THE DATASET
    # ==========================================
    # Load raw data from scikit-learn
    iris_raw = load_iris()

    # Convert into a Pandas DataFrame for better visualization
    df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    df["target"] = iris_raw.target

    print("--- Original DataFrame (First 5 Rows) ---")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\n" + "=" * 50 + "\n")

    # ==========================================
    # 2. CHECK AND HANDLE MISSING VALUES
    # ==========================================
    # Although the Iris dataset is clean, real-world data often has missing values (NaN)
    print("--- Missing Values Check ---")
    print(df.isnull().sum())

    # If missing values were present, you could handle them using:
    # df.fillna(df.mean(), inplace=True)  # Imputation
    # df.dropna(inplace=True)             # Deletion
    print("\nNo missing values found. Skipping imputation.\n")
    print("=" * 50 + "\n")

    # ==========================================
    # 3. SEPARATE FEATURES (X) AND TARGET (y)
    # ==========================================
    X = df.drop(columns=["target"])  # Features (Sepal/Petal lengths & widths)
    Y = df["target"]  # Target labels (0, 1, 2)

    # Optional: Map numeric labels back to descriptive string names for clarity
    target_names = iris_raw.target_names
    print("Target Classes:", target_names)

    # ==========================================
    # 4. FEATURE SCALING (STANDARDIZATION)
    # ==========================================
    # Standardization centers features to mean = 0 and variance = 1 (Z-score normalization).
    # This is crucial for distance-based algorithms like KNN, SVM, and Neural Networks.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame for readability
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    print("--- Scaled Features (First 5 Rows) ---")
    print(X_scaled_df.head())
    print(f"\nMean of scaled features (approx 0):\n{X_scaled_df.mean().round(2)}")
    print(f"Standard deviation of scaled features (1):\n{X_scaled_df.std().round(2)}")
    print("\n" + "=" * 50 + "\n")

    # ==========================================
    # 5. DATASET SPLITTING (TRAIN / TEST)
    # ==========================================
    # Split data into 80% training and 20% testing sets.
    # stratify=Y ensures the class distribution is preserved across both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print("--- Train-Test Split Dimensions ---")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train class distribution:\n", np.bincount(y_train))
    print("y_test class distribution:\n", np.bincount(y_test))
