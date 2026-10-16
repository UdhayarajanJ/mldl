# To Implement the Linear Regression algorithm using any dataset and draw a graph.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def lab_07():
    """
    Lab 07: Linear Regression Implementation and Visualization

    This lab demonstrates how to implement a simple linear regression model using a synthetic dataset.
    It covers generating data, splitting it into training and testing sets, training the model, evaluating
    its performance, and visualizing the results with a scatter plot and regression line.

    Steps:
    1. Generate a synthetic dataset (Study Hours vs Exam Scores).
    2. Split the dataset into training and testing sets.
    3. Train a linear regression model on the training data.
    4. Evaluate the model's performance using MSE, RMSE, and R-squared metrics.
    5. Visualize the training/testing data points along with the regression line.
    """
    # ==========================================
    # 1. GENERATE OR LOAD DATASET
    # ==========================================
    # For demonstration, we create a synthetic dataset (e.g., Study Hours vs Exam Scores)
    np.random.seed(42)
    study_hours = np.linspace(1, 10, 100)  # 1 to 10 hours
    # Linear relation: Score = 10 * hours + 20 + random noise
    exam_scores = 10 * study_hours + 20 + np.random.normal(0, 5, 100)

    # Reshape features to 2D array as required by Scikit-Learn
    X = study_hours.reshape(-1, 1)
    y = exam_scores

    print(f"Dataset Shape - Features (X): {X.shape}, Target (y): {y.shape}\n")

    # ==========================================
    # 2. SPLIT DATA INTO TRAIN & TEST SETS
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size:  {X_test.shape[0]} rows\n")

    # ==========================================
    # 3. TRAIN THE LINEAR REGRESSION MODEL
    # ==========================================
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Retrieve model parameters (slope/coefficient and intercept)
    print("--- Model Parameters ---")
    print(f"Coefficient (Slope 'm'): {model.coef_[0]:.4f}")
    print(f"Intercept ('c'):          {model.intercept_:.4f}")
    print(f"Equation: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}\n")

    # ==========================================
    # 4. EVALUATE MODEL PERFORMANCE
    # ==========================================
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("--- Performance Metrics ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared Score (R2): {r2:.4f} ({r2 * 100:.2f}% variance explained)\n")

    # ==========================================
    # 5. DRAW THE GRAPH (VISUALIZATION)
    # ==========================================
    plt.figure(figsize=(9, 6))

    # Plot actual training and testing points
    plt.scatter(X_train, y_train, color="blue", alpha=0.7, label="Training Data")
    plt.scatter(X_test, y_test, color="green", alpha=0.7, label="Testing Data")

    # Plot the regression line calculated by the model
    # We use the full range of X to draw a smooth continuous line
    plt.plot(X, model.predict(X), color="red", linewidth=2.5, label="Regression Line")

    # Graph styling and labels
    plt.title(
        "Linear Regression Fit (Study Hours vs Exam Score)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Study Hours", fontsize=12)
    plt.ylabel("Exam Score", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Display the plot
    plt.tight_layout()
    plt.show()
