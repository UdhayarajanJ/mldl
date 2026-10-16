# Implement Convolutional Neural Networks in Python with TensorFlow.

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


def lab_14():
    """
    Lab 14: Convolutional Neural Network (CNN) Implementation on Fashion MNIST Dataset

    This lab demonstrates how to implement a Convolutional Neural Network (CNN) using the Fashion MNIST dataset of clothing images.
    It covers loading and preprocessing the dataset, building the CNN architecture, training the model, evaluating its performance,
    and visualizing the training history.

    Steps:
    1. Load and preprocess the Fashion MNIST dataset.
    2. Build the CNN architecture with convolutional, pooling, and dense layers.
    3. Compile and train the CNN model on the training data.
    4. Evaluate the model's performance on the test data.
    5. Visualize training and validation accuracy and loss over epochs.
    """
    # ==========================================
    # 1. LOAD AND PREPROCESS THE DATASET
    # ==========================================
    # We use Fashion MNIST (70,000 grayscale images of 10 clothing categories)
    fashion_mnist = datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape:  {X_test.shape}\n")

    # Normalize pixel values from [0, 255] to [0.0, 1.0] for faster gradient descent convergence
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape data to include the channel dimension (28x28 pixels, 1 color channel for grayscale)
    X_train = X_train[..., tf.newaxis]
    X_test = X_test[..., tf.newaxis]

    # ==========================================
    # 2. BUILD THE CNN ARCHITECTURE
    # ==========================================
    model = models.Sequential(
        [
            # Feature Extraction Block 1
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            # Feature Extraction Block 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            # Feature Extraction Block 3
            layers.Conv2D(64, (3, 3), activation="relu"),
            # Classification Head (Dense Layers)
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(
                0.3
            ),  # Prevents overfitting by randomly dropping 30% of connections
            layers.Dense(10, activation="softmax"),  # 10 output nodes for 10 classes
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # Used when labels are integers rather than one-hot vectors
        metrics=["accuracy"],
    )

    # Print model summary showing layer dimensions and parameter counts
    model.summary()
    print("\n" + "=" * 50 + "\n")

    # ==========================================
    # 3. TRAIN THE MODEL
    # ==========================================
    # batch_size=64 processes 64 images per update; epochs=5 passes through dataset 5 times
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # ==========================================
    # 4. EVALUATE MODEL PERFORMANCE
    # ==========================================
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\n--- Model Evaluation on Test Data ---")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # ==========================================
    # 5. PLOT TRAINING HISTORY (Accuracy & Loss)
    # ==========================================
    plt.figure(figsize=(12, 4))

    # Accuracy Subplot
    plt.subplot(1, 2, 1)
    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="orange",
        linewidth=2,
    )
    plt.title("CNN Accuracy Over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss", color="red", linewidth=2)
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="purple",
        linewidth=2,
    )
    plt.title("CNN Loss Over Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
