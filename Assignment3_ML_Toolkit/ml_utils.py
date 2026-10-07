import numpy as np
import pandas as pd

def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Returns a summary of the dataset including number of rows, columns,
    total missing values, and duplicate rows.

    Parameters:
        df (pd.DataFrame): The input pandas DataFrame.

    Returns:
        dict: A dictionary containing 'rows', 'columns', 'missing_values',
              and 'duplicate_rows'.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().sum(),
        "duplicate_rows": df.duplicated().sum()
    }

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates missing value counts and percentages for each column.

    Parameters:
        df (pd.DataFrame): The input pandas DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with 'Missing Count' and 'Missing Percentage'.
    """
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    report = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })
    return report[report['Missing Count'] > 0]

def remove_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a cleaned copy of the DataFrame with rows containing missing values removed.

    Parameters:
        df (pd.DataFrame): The input pandas DataFrame.

    Returns:
        pd.DataFrame: A cleaned DataFrame.
    """
    return df.dropna().copy()

def normalize_array(values: np.ndarray) -> np.ndarray:
    """
    Performs Min-Max normalization on a numpy array.

    Parameters:
        values (np.ndarray): The input 1D numpy array.

    Returns:
        np.ndarray: Normalized array with values scaled between 0 and 1.
    """
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """
    Detects statistical outliers using the IQR method.

    Parameters:
        series (pd.Series): The input pandas Series containing numerical values.

    Returns:
        pd.Series: A boolean mask where True indicates an outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    return (series < lower_bound) | (series > upper_bound)
