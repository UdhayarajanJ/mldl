# Write a program to implement the Random Forest ensemble method on a given dataset.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def lab_09():
    """
    Lab 09: Random Forest Classifier Implementation and Feature Importance Visualization

    This lab demonstrates how to implement a Random Forest classifier using the Breast Cancer Wisconsin dataset.
    It covers loading the dataset, splitting it into training and testing sets, training the model, evaluating its performance,
    and visualizing the top feature importances.

    Steps:
    1. Load the Breast Cancer Wisconsin dataset.
    2. Split the dataset into training and testing sets.
    3. Train a Random Forest classifier on the training data.
    4. Evaluate the model's performance using accuracy, confusion matrix, and classification report.
    5. Visualize the top feature importances from the trained model.
    """
    # ==========================================
    # 1. LOAD DATASET
    # ==========================================
    # We use the Breast Cancer Wisconsin dataset for binary classification
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print(f"Dataset Shape - Features: {X.shape}, Target Classes: {data.target_names}\n")

    # ==========================================
    # 2. SPLIT DATA INTO TRAIN & TEST SETS
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 20% reserved for testing
        random_state=42,  # Ensures reproducible splits
        stratify=y,  # Maintains class distribution balance
    )

    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples:  {X_test.shape[0]}\n")

    # ==========================================
    # 3. TRAIN THE RANDOM FOREST MODEL
    # ==========================================
    # n_estimators=100 specifies building 100 individual decision trees in the forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None)

    # Train the model on the training data
    rf_model.fit(X_train, y_train)

    # ==========================================
    # 4. EVALUATE MODEL PERFORMANCE
    # ==========================================
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("--- Model Performance Metrics ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # ==========================================
    # 5. FEATURE IMPORTANCE VISUALIZATION
    # ==========================================
    # Random Forest provides feature importances based on impurity reduction across all trees
    importances = rf_model.feature_importances_
    feature_names = data.feature_names

    # Sort feature importances in descending order
    sorted_indices = np.argsort(importances)[::-1][:10]  # Top 10 features

    plt.figure(figsize=(10, 6))
    plt.title(
        "Top 10 Feature Importances (Random Forest)", fontsize=14, fontweight="bold"
    )
    plt.bar(
        range(10),
        importances[sorted_indices],
        align="center",
        color="forestgreen",
        edgecolor="black",
    )
    plt.xticks(range(10), feature_names[sorted_indices], rotation=45, ha="right")
    plt.ylabel("Importance Score", fontsize=12)
    plt.xlabel("Feature", fontsize=12)
    plt.tight_layout()

    # Display the feature importance graph
    plt.show()
