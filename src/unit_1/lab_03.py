# Write a program to implement a correlation matrix and draw a heatmap.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def lab_03():
    """
    Lab 03: Correlation Matrix and Heatmap Visualization

    This lab demonstrates how to compute a correlation matrix for a dataset
    and visualize it using a heatmap. It covers the following steps:
    1. Create or load a dataset.
    2. Compute the correlation matrix using Pandas.
    3. Visualize the correlation matrix as a heatmap using Seaborn.
    """
    # 1. Create a synthetic dataset (or load your own via pd.read_csv())
    np.random.seed(42)
    data = {
        "Study_Hours": [2, 3, 5, 6, 8, 10, 12],
        "Sleep_Hours": [8, 7.5, 7, 6.5, 6, 5.5, 5],
        "Exam_Score": [50, 55, 68, 72, 85, 92, 98],
        "Stress_Level": [3, 4, 5, 6, 7, 8, 9],
        "Attendance": [70, 75, 80, 85, 90, 95, 100],
    }

    # Convert dictionary into a Pandas DataFrame
    df = pd.DataFrame(data)

    print("--- Original Dataset ---")
    print(df)
    print("\n" + "=" * 50 + "\n")

    # 2. Compute the Correlation Matrix
    # By default, df.corr() uses Pearson's correlation coefficient
    corr_matrix = df.corr()

    print("--- Correlation Matrix ---")
    print(corr_matrix)
    print("\n" + "=" * 50 + "\n")

    # 3. Draw the Heatmap using Seaborn
    plt.figure(figsize=(8, 6))

    # sns.heatmap creates the matrix visualization
    sns.heatmap(
        corr_matrix,
        annot=True,  # Show the correlation values inside the cells
        cmap="coolwarm",  # Color palette (blue for negative, red for positive)
        fmt=".2f",  # Format numbers to 2 decimal places
        linewidths=0.5,  # Add lines between grid cells
        cbar=True,  # Show the color bar reference on the side
    )

    # Add title and layout adjustments
    plt.title("Correlation Matrix Heatmap", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Display the plot
    plt.show()
