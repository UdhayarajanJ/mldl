# To implement Logistic Regression by using any dataset and draw a graph.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def lab_08():
    """
    Lab 08: Logistic Regression Implementation and Visualization

    This lab demonstrates how to implement a logistic regression model using a synthetic binary classification dataset.
    It covers generating data, splitting it into training and testing sets, training the model, evaluating its performance,
    and visualizing the results with a scatter plot and the sigmoid decision curve.

    Steps:
    1. Generate a synthetic binary classification dataset.
    2. Split the dataset into training and testing sets.
    3. Train a logistic regression model on the training data.
    4. Evaluate the model's performance using accuracy and confusion matrix.
    5. Visualize the testing data points along with the sigmoid decision curve.
    """
    # ==========================================
    # 1. GENERATE A SYNTHETIC BINARY DATASET
    # ==========================================
    # We create a 2D dataset with 1 informative feature for clear 2D visualization
    X, y = make_classification(
        n_samples=200,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    print(f"Dataset Shape - Features (X): {X.shape}, Target (y): {y.shape}\n")

    # ==========================================
    # 2. SPLIT DATA INTO TRAIN & TEST SETS
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================================
    # 3. TRAIN THE LOGISTIC REGRESSION MODEL
    # ==========================================
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Model coefficients
    print("--- Model Parameters ---")
    print(f"Coefficient (Slope): {model.coef_[0][0]:.4f}")
    print(f"Intercept:           {model.intercept_[0]:.4f}\n")

    # ==========================================
    # 4. EVALUATE MODEL PERFORMANCE
    # ==========================================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("--- Performance Metrics ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ==========================================
    # 5. DRAW THE GRAPH (VISUALIZATION)
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Scatter plot of the data points (Class 0 and Class 1)
    plt.scatter(
        X_test[y_test == 0],
        y_test[y_test == 0],
        color="blue",
        label="Class 0 (Actual)",
        s=50,
        edgecolors="k",
    )
    plt.scatter(
        X_test[y_test == 1],
        y_test[y_test == 1],
        color="red",
        label="Class 1 (Actual)",
        s=50,
        edgecolors="k",
    )

    # Generate smooth values for plotting the Sigmoid curve
    X_curve = np.linspace(np.min(X), np.max(X), 300).reshape(-1, 1)
    # predict_proba returns probabilities for [class_0, class_1]; index [:, 1] is probability of class 1
    y_curve = model.predict_proba(X_curve)[:, 1]

    # Plot the Sigmoid decision curve
    plt.plot(
        X_curve, y_curve, color="green", linewidth=3, label="Sigmoid Decision Curve"
    )

    # Add threshold line at 0.5 probability
    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(1, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(0.5, color="orange", linestyle=":", label="Decision Threshold (0.5)")

    # Graph styling and labels
    plt.title(
        "Logistic Regression Sigmoid Curve & Classification",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Feature Value (X)", fontsize=12)
    plt.ylabel("Probability of Class 1", fontsize=12)
    plt.legend(loc="center right")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
