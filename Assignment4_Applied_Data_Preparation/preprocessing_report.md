# Applied Data Preparation Lab: Preprocessing Report

## 1. Introduction
This report documents the steps and decisions made during the data preparation phase of a healthcare machine learning assignment. The goal is to build a robust preprocessing pipeline to handle a messy dataset systematically without inadvertently leaking data or fabricating values.

## 2. Dataset Description
The dataset used is the **Diabetes 130-US Hospitals for Years 1999-2008** dataset. It contains patient admission information with various numerical and categorical features related to demographics, medical history, procedures, and length of stay. 
- Original shape: 101,766 rows and 50 columns.
- The target variable is `readmitted`.

## 3. Initial Data Problems
Upon inspection, several data quality issues were identified:
- **Missing Value Representation**: Missing values were represented as `'?'` strings instead of standard `NaN` values, requiring an explicit conversion step.
- **High Sparsity**: Certain columns like `weight` (96.8% missing) and `medical_specialty` (49% missing) had substantial missing data.
- **Inconsistent Categories**: Categorical values in the `gender` column contained entries like `Unknown/Invalid`, which behave conceptually as missing values and needed to be standardized.

## 4. Missing-Value Strategy
**Numerical Features:**  
Missing values in numerical features were handled using **median imputation** (`SimpleImputer(strategy='median')`). 
*Justification:* The median is robust to outliers, which are common in clinical variables (e.g., extremely high number of procedures or long hospital stays).

**Categorical Features:**  
Missing values in categorical features were imputed with a constant string **"Unknown"** (`SimpleImputer(strategy='constant', fill_value='Unknown')`). 
*Justification:* In healthcare data, missingness is often not random; the absence of a recorded value (like a payer code or a specific medical specialty) may carry predictive information itself. Therefore, preserving the missing state as a distinct category avoids losing this signal, unlike mode imputation which could introduce bias.

## 5. Categorical Encoding Strategy
Categorical variables were encoded using `OneHotEncoder(handle_unknown='ignore')`.
*Justification:* 
- One-hot encoding creates binary indicator columns for each category, which prevents the machine learning model from incorrectly inferring an ordinal relationship (e.g., race A < race B) that would occur with arbitrary integer encoding. 
- The `handle_unknown="ignore"` parameter is crucial because if the pipeline is deployed, it must gracefully handle unexpected new categories by simply outputting all zeros for that feature.

## 6. Numerical Scaling Strategy
Numerical features were scaled using `StandardScaler`, which transforms the data to have a mean of 0 and standard deviation of 1.
*Justification:* 
Variables like `time_in_hospital` (measured in days) and `num_lab_procedures` (counts) have vastly different numerical scales. Scaling is necessary to prevent features with intrinsically larger ranges from disproportionately influencing distance-based (e.g., KNN) or gradient-based algorithms (e.g., neural networks or logistic regression).

## 7. Feature Exclusion Decisions
The following columns were excluded from the input feature set:
- `encounter_id` and `patient_nbr`: These are administrative, non-predictive identifiers. Using them could cause the model to memorize specific patients rather than learning generalizable patterns.
- `readmitted`: This is the target variable. Including it would result in data leakage, where the model trains on the outcome it is supposed to predict.

## 8. Before/After Comparison

**BEFORE PREPROCESSING**
- Rows: 101,766
- Columns: 50
- Numerical columns: 8
- Categorical columns: 39

**AFTER PREPROCESSING**
- Rows: 101,766
- Columns: 2277
- Missing values: 0
- Categorical encoding: OneHotEncoder (columns expanded)
- Numerical scaling: StandardScaler (mean=0, variance=1)

## 9. Why a Pipeline was Used
A `scikit-learn` `Pipeline` combined with a `ColumnTransformer` was used instead of manually preprocessing each column.
*Justification:* 
Pipelines ensure reproducibility, bundle the steps together, and prevent data leakage during cross-validation by strictly applying transformations learned on the training set to the test set.

## 10. Limitations
- The current pipeline imputes heavily missing columns (like `weight`), which may just inject a large number of 'Unknown' categories that add dimensionality without strong predictive power.
- There are a massive number of unique diagnoses (`diag_1`, `diag_2`, `diag_3`) leading to very high dimensionality after one-hot encoding (expanding to >2000 features). Grouping these ICD-9 codes into broader clinical categories would be a more advanced but medically sound approach.

## 11. Conclusion
The dataset was successfully prepared using standard practices suited for messy healthcare data. By applying separate strategies for numerical and categorical features within a unified pipeline, the data is now fully transformed, scaled, encoded, and ready for machine learning algorithms.
