# Write a program to implement the Naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier and generate a confusion matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def lab_10():
    """
    Lab 10: Naïve Bayes Classifier Implementation and Evaluation

    This lab demonstrates how to implement a Naïve Bayes classifier using a sample dataset stored in a CSV file.
    It covers loading the dataset, preprocessing categorical features, splitting the data into training and testing sets,
    training the model, evaluating its performance, and visualizing the confusion matrix.

    Steps:
    1. Create or load a sample CSV dataset (e.g., Play Tennis).
    2. Preprocess categorical features into numerical format.
    3. Split the dataset into training and testing sets.
    4. Train a Gaussian Naïve Bayes classifier on the training data.
    5. Evaluate the model's performance using accuracy, confusion matrix, and classification report.
    6. Visualize the confusion matrix as a heatmap.
    """
    # ==========================================
    # 1. CREATE A SAMPLE CSV FILE (For Demonstration)
    # ==========================================
    # If you already have an existing CSV file (e.g., 'dataset.csv'),
    # you can skip this block and point pd.read_csv() to your file path.
    csv_data = """
                Outlook,Temperature,Humidity,Wind,PlayTennis
                Sunny,Hot,High,Weak,No
                Sunny,Hot,High,Strong,No
                Overcast,Hot,High,Weak,Yes
                Rain,Mild,High,Weak,Yes
                Rain,Cool,Normal,Weak,Yes
                Rain,Cool,Normal,Strong,No
                Overcast,Cool,Normal,Strong,Yes
                Sunny,Mild,High,Weak,No
                Sunny,Cool,Normal,Weak,Yes
                Rain,Mild,Normal,Weak,Yes
                Sunny,Mild,Normal,Strong,Yes
                Overcast,Mild,High,Strong,Yes
                Overcast,Hot,Normal,Weak,Yes
                Rain,Mild,High,Strong,No
                """

    # Write sample data to a file named 'play_tennis.csv'
    with open("data/play_tennis.csv", "w") as f:
        f.write(csv_data)

    print("Sample CSV file 'play_tennis.csv' created successfully.\n")

    # ==========================================
    # 2. LOAD DATA FROM CSV & PREPROCESS
    # ==========================================
    # Read the dataset using Pandas
    df = pd.read_csv("data/play_tennis.csv")

    print("--- First 5 Rows of CSV Dataset ---")
    print(df.head())
    print("\n" + "=" * 50 + "\n")

    # Naïve Bayes in Scikit-Learn expects numerical features.
    # We convert categorical text columns into numeric codes using factorize/get_dummies.
    df_encoded = df.apply(lambda col: col.factorize()[0])

    # Separate Features (X) and Target (y)
    # Assuming the last column is the target class
    X = df_encoded.iloc[:, :-1]
    y = df_encoded.iloc[:, -1]

    # ==========================================
    # 3. APPLY TRAIN-TEST SPLIT
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,  # 30% data reserved for testing
        random_state=42,  # Ensures reproducible splits
        stratify=y,  # Preserves class distribution proportions
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}\n")

    # ==========================================
    # 4. TRAIN THE NAÏVE BAYES CLASSIFIER
    # ==========================================
    # GaussianNB assumes continuous features follow a normal distribution.
    # For categorical/frequency data, CategoricalNB or MultinomialNB can also be used.
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = nb_classifier.predict(X_test)

    # ==========================================
    # 5. COMPUTE ACCURACY & EVALUATE PERFORMANCE
    # ==========================================
    accuracy = accuracy_score(y_test, y_pred)
    print("--- Evaluation Results ---")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ==========================================
    # 6. DRAW THE CONFUSION MATRIX HEATMAP
    # ==========================================
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.title(
        "Naïve Bayes Classifier - Confusion Matrix", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plt.show()
