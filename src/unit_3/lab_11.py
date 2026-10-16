# Build an Artificial Neural Network by using appropriate data sets

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def lab_11():
    """
    Lab 11: Artificial Neural Network (ANN) Implementation and Evaluation

    This lab demonstrates how to implement an Artificial Neural Network (ANN) using the Breast Cancer Wisconsin dataset.
    It covers loading the dataset, splitting it into training and testing sets, feature scaling, training the ANN model,
    evaluating its performance, and visualizing the training loss curve.

    Steps:
    1. Load the Breast Cancer Wisconsin dataset.
    2. Split the dataset into training and testing sets.
    3. Scale features to ensure proper convergence during training.
    4. Build and train an ANN model on the scaled training data.
    5. Evaluate the model's performance using accuracy, confusion matrix, and classification report.
    6. Visualize the training loss curve to assess convergence over epochs.
    """
    # ==========================================
    # 1. LOAD DATASET
    # ==========================================
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
        random_state=42,  # Ensures reproducibility
        stratify=y,  # Preserves class balance
    )

    # ==========================================
    # 3. FEATURE SCALING (Crucial for Neural Networks)
    # ==========================================
    # ANNs rely on gradient descent optimization; scaling features to mean=0, variance=1
    # prevents variables with large numeric ranges from dominating weight updates.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 4. BUILD & TRAIN THE ARTIFICIAL NEURAL NETWORK
    # ==========================================
    # hidden_layer_sizes=(64, 32): Two hidden layers with 64 and 32 neurons respectively
    # activation='relu': Rectified Linear Unit activation function
    # max_iter=500: Maximum epochs for training convergence
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    # Train the network
    mlp_model.fit(X_train_scaled, y_train)

    # ==========================================
    # 5. EVALUATE MODEL PERFORMANCE
    # ==========================================
    y_pred = mlp_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print("--- Model Performance Metrics ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # ==========================================
    # 6. PLOT TRAINING LOSS CURVE
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.plot(mlp_model.loss_curve_, color="purple", linewidth=2)
    plt.title(
        "ANN Training Loss Curve (Convergence over Epochs)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Epoch / Iteration", fontsize=11)
    plt.ylabel("Loss Value", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
