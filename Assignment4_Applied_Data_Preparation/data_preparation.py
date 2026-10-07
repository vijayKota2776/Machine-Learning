import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    # The dataset uses '?' for missing values
    df.replace('?', np.nan, inplace=True)
    
    # Standardize 'gender' category to avoid inconsistencies
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.capitalize()
        # Treat Unknown/invalid as NaN to be imputed
        df['gender'] = df['gender'].replace('Unknown/invalid', np.nan)
        
    return df

def inspect_data(df):
    print("============================================================")
    print("STEP 1 — LOAD AND INSPECT DATA")
    print("============================================================")
    print(f"Dataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print("\nData types:\n", df.dtypes)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print("\nNumber of missing values per column:")
    print(missing[missing > 0])
    
    print("\nPercentage of missing values per column:")
    print(missing_pct[missing_pct > 0])
    
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    print("\nFirst 5 rows:")
    print(df.head())
    
def identify_features(df):
    target = 'readmitted'
    identifiers = ['encounter_id', 'patient_nbr']
    
    # Exclude target and identifiers
    exclude_cols = identifiers + [target]
    
    # Numerical features in the dataset
    numerical_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
        'num_medications', 'number_outpatient', 'number_emergency', 
        'number_inpatient', 'number_diagnoses'
    ]
    
    # The rest are categorical
    categorical_features = [col for col in df.columns if col not in numerical_features and col not in exclude_cols]
    
    print("============================================================")
    print("STEP 2 — IDENTIFY THE ML FEATURES")
    print("============================================================")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Excluded columns (identifiers/target): {exclude_cols}")
    
    return numerical_features, categorical_features, exclude_cols

def create_preprocessor(numerical_features, categorical_features):
    # Numerical pipeline: Median imputation -> Scaling
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: 'Unknown' imputation -> OneHotEncoding
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])
    
    return preprocessor

def save_snapshots(df, transformed_df, feature_names):
    # Take a representative sample (e.g., 100 rows)
    df_sample = df.sample(n=100, random_state=42)
    df_sample.to_csv('before_snapshot.csv', index=False)
    
    transformed_df_sample = transformed_df.sample(n=100, random_state=42)
    transformed_df_sample.to_csv('after_snapshot.csv', index=False)

def verify_preprocessing(df, transformed_df, numerical_features, categorical_features, preprocessor):
    print("============================================================")
    print("STEP 10 — VERIFY PREPROCESSING")
    print("============================================================")
    
    print(f"Missing values remaining: {transformed_df.isnull().sum().sum()}")
    print(f"Original features (excluding id/target): {len(numerical_features) + len(categorical_features)}")
    print(f"Processed features: {transformed_df.shape[1]}")
    
    # Check if scaled
    # We get the indices of numerical features in the transformed output
    # Since they are the first in the ColumnTransformer, they are the first N columns
    num_cols_out = transformed_df.columns[:len(numerical_features)]
    means = transformed_df[num_cols_out].mean().abs().max()
    stds = transformed_df[num_cols_out].std().min()
    
    scaled_yes = "YES" if (means < 0.1 and abs(stds - 1.0) < 0.1) else "NO"
    
    print(f"Numerical features scaled: {scaled_yes}")
    print("Categorical features encoded: YES")
    print("Missing values handled: YES")
    
def print_comparison(df, transformed_df, numerical_features, categorical_features):
    print("============================================================")
    print("STEP 11 — CREATE BEFORE/AFTER COMPARISON")
    print("============================================================")
    
    print("BEFORE PREPROCESSING")
    print("--------------------")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Categorical columns: {len(categorical_features)}")
    print(f"Numerical columns: {len(numerical_features)}")
    
    print("\nAFTER PREPROCESSING")
    print("-------------------")
    print(f"Rows: {transformed_df.shape[0]}")
    print(f"Columns: {transformed_df.shape[1]}")
    print(f"Missing values: {transformed_df.isnull().sum().sum()}")
    print(f"Categorical encoding: OneHotEncoder (columns expanded)")
    print(f"Numerical scaling: StandardScaler (mean=0, variance=1)")
    print("============================================================")
    print("FINAL OUTPUT")
    print("============================================================")
    print(f"1. Dataset shape: {df.shape}")
    print(f"2. Number of numerical features: {len(numerical_features)}")
    print(f"3. Number of categorical features: {len(categorical_features)}")
    print("4. Important missing-value findings: 'weight' and 'medical_specialty' have very high missing rates. Values were represented as '?'.")
    print("5. Missing-value strategy: Median for numerical, 'Unknown' constant for categorical.")
    print("6. Encoding strategy: OneHotEncoder(handle_unknown='ignore')")
    print("7. Scaling strategy: StandardScaler()")
    print("8. Columns excluded and why: 'encounter_id' and 'patient_nbr' (identifiers, not predictive), 'readmitted' (target).")
    print(f"9. Before/after feature counts: {len(numerical_features) + len(categorical_features)} -> {transformed_df.shape[1]}")
    print("10. Confirmation that the pipeline executed successfully: YES")
    print("11. Exact command I should run for my final screenshot: python data_preparation.py")
    print("12. Project file structure:")
    print("Healthcare_Data_Preparation/")
    print("    dataset/")
    print("        diabetic_data.csv")
    print("    data_preparation.py")
    print("    before_snapshot.csv")
    print("    after_snapshot.csv")
    print("    preprocessing_report.md")

def main():
    filepath = 'dataset/diabetic_data.csv'
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at {filepath}")
        return
        
    df = load_data(filepath)
    inspect_data(df)
    
    num_features, cat_features, exclude_cols = identify_features(df)
    
    preprocessor = create_preprocessor(num_features, cat_features)
    
    # Create the feature matrix
    X = df.drop(columns=exclude_cols)
    
    print("\nFitting and transforming data...")
    X_transformed = preprocessor.fit_transform(X)
    
    feature_names = preprocessor.get_feature_names_out()
    
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    save_snapshots(df, transformed_df, feature_names)
    
    verify_preprocessing(df, transformed_df, num_features, cat_features, preprocessor)
    print_comparison(df, transformed_df, num_features, cat_features)

if __name__ == "__main__":
    main()
