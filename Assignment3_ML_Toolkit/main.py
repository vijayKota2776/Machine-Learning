import numpy as np
import pandas as pd
from ml_utils import (
    dataset_summary,
    missing_value_report,
    remove_missing_rows,
    normalize_array,
    detect_outliers_iqr
)
import matplotlib
import sklearn

def main():
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("All ML libraries imported successfully!\n")

    # Sample dataset
    data = {
        'Age': [25, 30, np.nan, 22, 100, 25],
        'Salary': [50000, 60000, 55000, np.nan, 200000, 50000],
        'Department': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT']
    }
    df = pd.DataFrame(data)

    print("--- 1. Original Dataset ---")
    print(df)
    print("\n")

    print("--- 2. Dataset Summary ---")
    summary = dataset_summary(df)
    for key, val in summary.items():
        print(f"{key.replace('_', ' ').title()}: {val}")
    print("\n")

    print("--- 3. Missing Value Report ---")
    report = missing_value_report(df)
    print(report)
    print("\n")

    print("--- 4. Result After Missing-Value Removal ---")
    cleaned_df = remove_missing_rows(df)
    print(cleaned_df)
    print("\n")

    print("--- 5. Normalized Numerical Values ---")
    ages = df['Age'].dropna().values
    normalized_ages = normalize_array(ages)
    print("Original Ages (Cleaned):", ages)
    print("Normalized Ages:", normalized_ages)
    print("\n")

    print("--- 6. Detected Outliers ---")
    # Using 'Salary' after removing NaNs for outlier detection
    salaries = df['Salary'].dropna()
    outlier_mask = detect_outliers_iqr(salaries)
    print("Salaries:\n", salaries)
    print("Outlier Mask:\n", outlier_mask)
    print("Outliers Found:\n", salaries[outlier_mask])
    print("\n")

    print("UTILITY MODULE EXECUTED SUCCESSFULLY")

if __name__ == "__main__":
    main()
