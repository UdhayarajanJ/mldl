# Import the diabetes dataset in .csv format and calculate the accuracy of a given model by using various feature selection algorithms

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Feature Selection Modules
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_selection import SelectFromModel


def lab_12():
    """
    Lab 12: Feature Selection Techniques and Model Evaluation

    This lab demonstrates how to apply various feature selection techniques on a dataset
    and evaluate the performance of a machine learning model using the selected features.
    It covers loading a sample dataset, preprocessing, applying feature selection methods,
    training a Random Forest classifier, and comparing the accuracy of models built with
    different feature subsets.

    Steps:
    1. Create or load a sample dataset (e.g., diabetes dataset).
    2. Preprocess the data (train-test split and feature scaling).
    3. Train a baseline Random Forest model using all features.
    4. Apply different feature selection techniques:
       - SelectKBest
       - Recursive Feature Elimination (RFE)
       - SelectFromModel
    5. Train models using the selected features and evaluate their performance.
    """
    # ==========================================
    # 1. CREATE A SAMPLE DIABETES CSV FILE
    # ==========================================
    # If you have an existing file (e.g., 'diabetes.csv'), skip this block
    # and point pd.read_csv() directly to your file path.
    csv_content = """   Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
                        6,148,72,35,0,33.6,0.627,50,1
                        1,85,66,29,0,26.6,0.351,31,0
                        8,183,64,0,0,23.3,0.672,32,1
                        1,89,66,23,94,28.1,0.167,21,0
                        0,137,40,35,168,43.1,2.288,33,1
                        5,116,74,0,0,25.6,0.201,30,0
                        3,78,50,32,88,31.0,0.248,26,1
                        10,115,0,0,0,35.3,0.134,29,0
                        2,197,70,45,543,30.5,0.158,53,1
                        8,125,96,0,0,0.0,0.232,54,1
                        4,110,92,0,0,37.6,0.191,30,0
                        10,168,74,0,0,38.0,0.537,34,1
                        10,139,80,0,0,27.1,1.441,57,0
                        1,189,60,23,846,30.1,0.398,59,1
                    """

    with open("data/diabetes_sample.csv", "w") as f:
        f.write(csv_content)

    print("Sample CSV file 'diabetes_sample.csv' created.\n")

    # ==========================================
    # 2. LOAD DATA AND PREPROCESS
    # ==========================================
    df = pd.read_csv("data/diabetes_sample.csv")

    # Separate Features (X) and Target (Outcome: 0 or 1)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 3. BASE MODEL (All Features)
    # ==========================================
    base_model = RandomForestClassifier(random_state=42)
    base_model.fit(X_train_scaled, y_train)
    y_pred_base = base_model.predict(X_test_scaled)
    base_accuracy = accuracy_score(y_test, y_pred_base)

    print("--- Baseline Performance ---")
    print(
        f"Accuracy using ALL features ({X.shape[1]} features): {base_accuracy * 100:.2f}%\n"
    )

    # ==========================================
    # 4. FEATURE SELECTION ALGORITHMS & EVALUATION
    # ==========================================
    print("--- Evaluating Feature Selection Methods ---")

    # Method A: SelectKBest (Statistical ANOVA F-value)
    # Selects top K features (e.g., K=4)
    selector_kbest = SelectKBest(score_func=f_classif, k=4)
    X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
    X_test_kbest = selector_kbest.transform(X_test_scaled)

    model_kbest = RandomForestClassifier(random_state=42)
    model_kbest.fit(X_train_kbest, y_train)
    acc_kbest = accuracy_score(y_test, model_kbest.predict(X_test_kbest))
    print(f"1. SelectKBest (Top 4 Features): Accuracy = {acc_kbest * 100:.2f}%")

    # Method B: Recursive Feature Elimination (RFE)
    # Recursively removes weakest features using an estimator
    estimator = RandomForestClassifier(random_state=42)
    rfe = RFE(estimator=estimator, n_features_to_select=4, step=1)
    X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
    X_test_rfe = rfe.transform(X_test_scaled)

    model_rfe = RandomForestClassifier(random_state=42)
    model_rfe.fit(X_train_rfe, y_train)
    acc_rfe = accuracy_score(y_test, model_rfe.predict(X_test_rfe))
    print(f"2. RFE (Top 4 Features):         Accuracy = {acc_rfe * 100:.2f}%")

    # Method C: SelectFromModel (Tree-based Importance Threshold)
    # Selects features based on feature importance weights from Random Forest
    embedded_selector = SelectFromModel(
        RandomForestClassifier(random_state=42), threshold="median"
    )
    X_train_sfm = embedded_selector.fit_transform(X_train_scaled, y_train)
    X_test_sfm = embedded_selector.transform(X_test_scaled)

    model_sfm = RandomForestClassifier(random_state=42)
    model_sfm.fit(X_train_sfm, y_train)
    acc_sfm = accuracy_score(y_test, model_sfm.predict(X_test_sfm))
    print(f"3. SelectFromModel (Median Thresh): Accuracy = {acc_sfm * 100:.2f}%")
