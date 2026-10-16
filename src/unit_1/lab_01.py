# Write a Python program to import the NumPy library and implement basic functions.

import numpy as np


def lab_01():
    """
    This lab demonstrates the basic functionalities of the NumPy library.
    It covers array creation, attributes, arithmetic operations, statistical functions,
    and indexing/slicing of arrays.
    Steps:
    1. Create different types of NumPy arrays (1D, 2D, zeros, ones, range).
    2. Explore array attributes such as shape, size, and data type.
    3. Perform basic arithmetic operations and use universal mathematical functions.
    4. Calculate statistical measures like mean, median, standard deviation, etc.
    5. Demonstrate indexing and slicing of arrays.
    """
    print("=== 1. ARRAY CREATION ===")

    # Creating a 1D array
    arr_1d = np.array([10, 20, 30, 40, 50])
    print("1D Array:", arr_1d)

    # Creating a 2D array (Matrix)
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print("2D Array:\n", arr_2d)

    # Creating special arrays
    zeros_arr = np.zeros((2, 3))  # 2x3 matrix of zeros
    ones_arr = np.ones((3, 2))  # 3x2 matrix of ones
    range_arr = np.arange(0, 10, 2)  # Values from 0 to 10 with step 2

    print("Zeros Array (2x3):\n", zeros_arr)
    print("Ones Array (3x2):\n", ones_arr)
    print("Range Array (0 to 10, step 2):", range_arr)

    print("\n=== 2. ARRAY ATTRIBUTES ===")

    print("Shape of 2D Array (rows, cols):", arr_2d.shape)
    print("Number of Dimensions (ndim):", arr_2d.ndim)
    print("Total Number of Elements (size):", arr_2d.size)
    print("Data Type of Elements (dtype):", arr_2d.dtype)

    print("\n=== 3. BASIC ARITHMETIC & MATHEMATICAL FUNCTIONS ===")

    a = np.array([10, 20, 30])
    b = np.array([1, 2, 3])

    print("Array a:", a)
    print("Array b:", b)

    # Element-wise operations
    print("Addition (a + b):", a + b)
    print("Subtraction (a - b):", a - b)
    print("Multiplication (a * b):", a * b)

    # Universal mathematical functions
    print("Square root of a:", np.sqrt(a))
    print("Exponential of b:", np.exp(b))
    print("Sine values of a:", np.sin(a))

    print("\n=== 4. STATISTICAL FUNCTIONS ===")

    scores = np.array([75, 88, 92, 60, 85, 90])
    print("Scores Data:", scores)

    print("Mean (Average):", np.mean(scores))
    print("Median:", np.median(scores))
    print("Standard Deviation:", np.std(scores))
    print("Sum of all scores:", np.sum(scores))
    print("Maximum Score:", np.max(scores))
    print("Minimum Score:", np.min(scores))

    print("\n=== 5. INDEXING AND SLICING ===")

    matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

    print("Original 3x3 Matrix:\n", matrix)

    # Accessing a specific element (row 1, column 2 -> 60)
    print("Element at row 1, col 2:", matrix[1, 2])

    # Slicing rows and columns (First 2 rows, all columns)
    print("First two rows:\n", matrix[:2, :])

    # Slicing specific columns (All rows, column 1 onwards)
    print("Columns from index 1 onwards:\n", matrix[:, 1:])
