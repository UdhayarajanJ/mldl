# Write a program to evaluate the performance measure of any classifier. Apply Train-Test Split for Evaluating Machine Learning Algorithms.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def lab_06():
    """
    Lab 06: Classifier Performance Evaluation with Train-Test Split

    This lab demonstrates how to evaluate the performance of a machine learning classifier
    using the Iris dataset. It covers loading the dataset, splitting it into training and
    testing sets, training a K-Nearest Neighbors (KNN) classifier, making predictions, and
    calculating various performance metrics.

    Steps:
    1. Load the Iris dataset.
    2. Split the dataset into training and testing sets using train_test_split.
    3. Train a K-Nearest Neighbors (KNN) classifier on the training data.
    4. Make predictions on the test data.
    5. Evaluate the classifier's performance using accuracy, precision, recall, F1-score,
       confusion matrix, and a detailed classification report.
    """

    # ==========================================
    # 1. LOAD DATASET
    # ==========================================
    iris = load_iris()
    X = iris.data  # Features (Sepal & Petal lengths/widths)
    y = iris.target  # Target classes (0, 1, 2)

    print(f"Total dataset shape - Features: {X.shape}, Labels: {y.shape}\n")

    # ==========================================
    # 2. APPLY TRAIN-TEST SPLIT
    # ==========================================
    # Split data into 80% training and 20% testing sets.
    # stratify=y ensures each set preserves the same proportion of target classes.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 20% for testing
        random_state=42,  # Ensures reproducible splits
        stratify=y,  # Maintains class distribution balance
    )

    print("--- Train-Test Split Dimensions ---")
    print(f"X_train (Training Features): {X_train.shape}")
    print(f"X_test  (Testing Features):  {X_test.shape}")
    print(f"y_train (Training Labels):   {y_train.shape}")
    print(f"y_test  (Testing Labels):    {y_test.shape}\n")

    # ==========================================
    # 3. TRAIN THE CLASSIFIER
    # ==========================================
    # Initialize and train a K-Nearest Neighbors Classifier
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    # Predict on the unseen test data
    y_pred = classifier.predict(X_test)

    # ==========================================
    # 4. EVALUATE PERFORMANCE MEASURES
    # ==========================================
    print("=" * 50)
    print("       CLASSIFIER PERFORMANCE METRICS")
    print("=" * 50)

    # A. Accuracy Score (Correct predictions / Total predictions)
    acc = accuracy_score(y_test, y_pred)
    print(f"1. Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")

    # B. Precision, Recall, and F1-Score (using 'macro' averaging for multi-class)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"2. Precision: {precision:.4f}")
    print(f"3. Recall:    {recall:.4f}")
    print(f"4. F1-Score:  {f1:.4f}\n")

    # C. Confusion Matrix
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("\n(Rows: Actual Classes, Columns: Predicted Classes)\n")

    # D. Detailed Classification Report
    print("--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
