# Write a Python program to import and export data using Pandas library functions.

import pandas as pd


def lab_02():
    """
    Lab 02: Data Export and Import with Pandas

    This lab demonstrates how to export and import data using Pandas.
    It covers exporting data to CSV, JSON, and Excel formats, as well as
    importing data from these formats back into a Pandas DataFrame.

    Steps:
    1. Create a sample DataFrame.
    2. Export the DataFrame to CSV, JSON, and Excel files.
    3. Import the data back from these files into new DataFrames.
    """
    # ==========================================
    # 1. CREATE A SAMPLE DATAFRAME
    # ==========================================
    data = {
        "Student_ID": [101, 102, 103, 104, 105],
        "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan"],
        "Score": [85, 92, 78, 95, 88],
        "Passed": [True, True, True, True, True],
    }

    # Convert dictionary to a Pandas DataFrame
    df = pd.DataFrame(data)
    print("--- Original DataFrame ---")
    print(df)
    print("\n" + "=" * 40 + "\n")

    # ==========================================
    # 2. EXPORTING DATA
    # ==========================================

    # A. Export to CSV (Comma-Separated Values)
    # index=False prevents Pandas from writing row numbers into the file
    df.to_csv("data/students_data.csv", index=False)
    print("Data successfully exported to 'students_data.csv'")

    # B. Export to JSON (JavaScript Object Notation)
    df.to_json("data/students_data.json", orient="records", indent=4)
    print("Data successfully exported to 'students_data.json'")

    # C. Export to Excel
    # Note: Requires 'openpyxl' library installed (pip install openpyxl)
    try:
        df.to_excel("data/students_data.xlsx", index=False, sheet_name="Results")
        print("Data successfully exported to 'students_data.xlsx'")
    except Exception as e:
        print("Excel export skipped (requires openpyxl):", e)

    print("\n" + "=" * 40 + "\n")

    # ==========================================
    # 3. IMPORTING DATA
    # ==========================================

    # A. Import from CSV
    print("--- Importing from CSV ---")
    df_from_csv = pd.read_csv("data/students_data.csv")
    print(df_from_csv)
    print("\n")

    # B. Import from JSON
    print("--- Importing from JSON ---")
    df_from_json = pd.read_json("data/students_data.json")
    print(df_from_json)
    print("\n")

    # C. Import from Excel
    try:
        print("--- Importing from Excel ---")
        df_from_excel = pd.read_excel("data/students_data.xlsx", sheet_name="Results")
        print(df_from_excel)
    except Exception as e:
        print("Excel import skipped:", e)
