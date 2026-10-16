# Implement Convolutional Neural Networks in Python with Keras.

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


def lab_13():
    """
    Lab 13: Convolutional Neural Network (CNN) Implementation on MNIST Dataset

    This lab demonstrates how to implement a Convolutional Neural Network (CNN) using the MNIST dataset of handwritten digits.
    It covers loading and preprocessing the dataset, building the CNN architecture, training the model, evaluating its performance,
    and visualizing the training history.

    Steps:
    1. Load and preprocess the MNIST dataset.
    2. Build the CNN architecture with convolutional, pooling, and dense layers.
    3. Compile and train the CNN model on the training data.
    4. Evaluate the model's performance on the test data.
    5. Visualize training and validation accuracy and loss over epochs.
    """

    # ==========================================
    # 1. LOAD AND PREPROCESS THE MNIST DATASET
    # ==========================================
    # Load training and testing data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(f"Original Training Data Shape: {X_train.shape}")
    print(f"Original Testing Data Shape:  {X_test.shape}\n")

    # Reshape data to include channel dimension (28x28 pixels, 1 color channel for grayscale)
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype("float32")
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype("float32")

    # Normalize pixel values to be between 0 and 1 (improves gradient descent convergence)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # One-hot encode target labels (e.g., class 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    y_train_encoded = to_categorical(y_train, 10)
    y_test_encoded = to_categorical(y_test, 10)

    # ==========================================
    # 2. BUILD THE CNN ARCHITECTURE
    # ==========================================
    model = Sequential(
        [
            # Convolutional Layer 1: 32 filters, 3x3 kernel, ReLU activation
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            # Max Pooling Layer 1: Reduces spatial dimensions (2x2 pool size)
            MaxPooling2D(pool_size=(2, 2)),
            # Convolutional Layer 2: 64 filters, 3x3 kernel, ReLU activation
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # Max Pooling Layer 2
            MaxPooling2D(pool_size=(2, 2)),
            # Flatten layer to convert 2D feature maps into 1D vectors for dense layers
            Flatten(),
            # Fully Connected Dense Layer with Dropout for regularization
            Dense(128, activation="relu"),
            Dropout(
                0.5
            ),  # Drops 50% of neurons randomly during training to prevent overfitting
            # Output Layer: 10 units for digits 0-9 with Softmax activation for probabilities
            Dense(10, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()
    print("\n" + "=" * 50 + "\n")

    # ==========================================
    # 3. TRAIN THE CNN MODEL
    # ==========================================
    # batch_size=64 processes 64 images per gradient update; epochs=5 passes over dataset 5 times
    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(X_test, y_test_encoded),
        epochs=5,
        batch_size=64,
        verbose=1,
    )

    # ==========================================
    # 4. EVALUATE MODEL PERFORMANCE
    # ==========================================
    test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
    print("\n--- Model Evaluation ---")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

    # ==========================================
    # 5. PLOT TRAINING HISTORY (Accuracy & Loss)
    # ==========================================
    plt.figure(figsize=(12, 4))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="blue")
    plt.plot(
        history.history["val_accuracy"], label="Validation Accuracy", color="orange"
    )
    plt.title("CNN Training & Validation Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss", color="red")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="purple")
    plt.title("CNN Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
