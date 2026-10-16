# Write a Python program to demonstrate various Data Visualization Techniques.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def lab_05():
    """
    Lab 05: Data Visualization Techniques

    This lab demonstrates various data visualization techniques using Matplotlib and Seaborn.
    It covers line plots, bar charts, histograms, scatter plots, box plots, and heatmaps.

    Steps:
    1. Set a clean aesthetic style for the plots.
    2. Create sample datasets for demonstration.
    3. Generate a 2x3 subplot grid to showcase different visualization techniques.
    4. Customize each plot with titles, labels, and legends as necessary.
    5. Display the complete dashboard of visualizations.
    """
    # Set a clean aesthetic style for the plots
    sns.set_theme(style="whitegrid")

    # Create a sample dataset for demonstration
    np.random.seed(42)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    sales = [120, 150, 170, 130, 200, 220]
    categories = ["Electronics", "Clothing", "Home", "Books", "Sports"]
    category_sales = [450, 320, 280, 150, 390]

    # Generate synthetic numerical data for distributions and correlations
    data_size = 200
    df = pd.DataFrame(
        {
            "Study_Hours": np.random.normal(6, 1.5, data_size),
            "Sleep_Hours": np.random.normal(7, 1.0, data_size),
            "Exam_Score": np.random.randint(50, 100, data_size),
            "Performance_Level": np.random.choice(["Low", "Medium", "High"], data_size),
        }
    )

    # Initialize a 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Core Data Visualization Techniques Suite",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    # ----------------------------------------------------
    # 1. LINE PLOT (Trends over time/sequence)
    # ----------------------------------------------------
    axes[0, 0].plot(months, sales, marker="o", color="b", linewidth=2.5, markersize=8)
    axes[0, 0].set_title("1. Line Plot (Sales Trend)")
    axes[0, 0].set_xlabel("Months")
    axes[0, 0].set_ylabel("Sales ($)")

    # ----------------------------------------------------
    # 2. BAR CHART (Comparing discrete categories)
    # ----------------------------------------------------
    axes[0, 1].bar(categories, category_sales, color="skyblue", edgecolor="black")
    axes[0, 1].set_title("2. Bar Chart (Category Comparison)")
    axes[0, 1].set_xlabel("Categories")
    axes[0, 1].set_ylabel("Revenue")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # ----------------------------------------------------
    # 3. HISTOGRAM (Showing data distribution)
    # ----------------------------------------------------
    axes[0, 2].hist(
        df["Study_Hours"], bins=15, color="orange", edgecolor="black", alpha=0.7
    )
    axes[0, 2].set_title("3. Histogram (Study Hours Distribution)")
    axes[0, 2].set_xlabel("Hours")
    axes[0, 2].set_ylabel("Frequency")

    # ----------------------------------------------------
    # 4. SCATTER PLOT (Relationship between two variables)
    # ----------------------------------------------------
    axes[1, 0].scatter(
        df["Study_Hours"], df["Exam_Score"], color="green", alpha=0.6, edgecolors="w"
    )
    axes[1, 0].set_title("4. Scatter Plot (Hours vs. Score)")
    axes[1, 0].set_xlabel("Study Hours")
    axes[1, 0].set_ylabel("Exam Score")

    # ----------------------------------------------------
    # 5. BOX PLOT (Detecting spread and outliers)
    # ----------------------------------------------------
    sns.boxplot(
        ax=axes[1, 1], x="Performance_Level", y="Sleep_Hours", data=df, palette="Set2"
    )
    axes[1, 1].set_title("5. Box Plot (Sleep by Performance)")
    axes[1, 1].set_xlabel("Performance Level")
    axes[1, 1].set_ylabel("Sleep Hours")

    # ----------------------------------------------------
    # 6. HEATMAP (Matrix correlation visualization)
    # ----------------------------------------------------
    corr = df[["Study_Hours", "Sleep_Hours", "Exam_Score"]].corr()
    sns.heatmap(
        ax=axes[1, 2], data=corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=False
    )
    axes[1, 2].set_title("6. Heatmap (Feature Correlation)")

    # Adjust layout for clean spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # Display the complete dashboard
    plt.show()
