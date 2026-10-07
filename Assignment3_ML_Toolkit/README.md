# ML Toolkit Setup

## Objective
This project demonstrates environment setup, Python modules, reusable functions, imports, and basic ML utilities without training a synthetic machine learning model.

## Python Version
Built with Python 3 (use `python3 --version` to verify).

## Required Packages
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Environment Setup
To create and set up the virtual environment:
1. Open your terminal.
2. Run the following command to create the virtual environment:
   ```bash
   python3 -m venv .venv
   ```
3. Activate the environment:
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
4. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

## Running the Project
After setting up and activating the environment, run the main script:
```bash
python main.py
```

## Utility Functions (`ml_utils.py`)
1. `dataset_summary(df)`: Returns the number of rows, columns, total missing values, and duplicate row count.
2. `missing_value_report(df)`: Calculates missing-value counts and percentages, returning them in a DataFrame.
3. `remove_missing_rows(df)`: Returns a cleaned copy of a DataFrame by removing rows with missing values.
4. `normalize_array(values)`: Performs Min-Max normalization on a NumPy array.
5. `detect_outliers_iqr(series)`: Detects statistical outliers using the IQR method and returns a boolean mask.

## Project Structure
```
ML_Toolkit_Setup/
├── .venv/               # Virtual environment directory (git-ignored)
├── ml_utils.py          # Reusable utility functions
├── main.py              # Main script testing utilities
└── README.md            # Project documentation
```
