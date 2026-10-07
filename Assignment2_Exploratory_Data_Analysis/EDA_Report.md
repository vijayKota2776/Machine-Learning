# Healthcare Exploratory Data Analysis Report

**Title:** Exploratory Data Analysis on Hospital Patient Readmission Data
**Dataset:** Diabetes 130-US Hospitals for Years 1999-2008

## Overview
This report summarizes the findings from the Exploratory Data Analysis (EDA) conducted on the Diabetes 130-US Hospitals dataset. The analysis aimed to profile the dataset, identify missing values and outliers, and determine suitable machine learning problem types.

## Key Findings

1. **Dataset Size:** The dataset contains exactly 101,766 records (patient encounters) and 50 features.
2. **Missing Values:** There is significant missing data in key clinical columns. Specifically, `weight` is missing in 96.8% (98,569) of cases, and `medical_specialty` is missing in 49% (49,949) of cases. These values are coded as `?`.
3. **Outliers:** Based on the Interquartile Range (IQR) method, the dataset contains several potential statistical outliers in crucial features:
    * `num_medications`: 2,557 outliers
    * `time_in_hospital`: 2,252 outliers
    * *Note:* In the healthcare context, these outliers likely represent clinically severe cases rather than data-entry errors.
4. **Target Imbalance:** The readmission target is heavily imbalanced:
    * **NO Readmission:** 54,864 encounters (53.91%)
    * **> 30 Days Readmission:** 35,545 encounters (34.93%)
    * **< 30 Days Readmission:** 11,357 encounters (11.16%)
5. **Relationships:** A clear positive correlation (0.47) exists between `time_in_hospital` and `num_medications`.
6. **Demographics:** The dataset heavily overrepresents Caucasian patients and those in the 50-90 age brackets, pointing to potential representation bias.

## Machine Learning Suitability

Based on the profiling, the dataset is exceptionally well-suited for multiple Machine Learning tasks.

**1. Classification:** 
* **Supported:** Yes.
* **Problem:** Predicting whether a diabetic patient will be readmitted to the hospital within 30 days (`<30`), after 30 days (`>30`), or not at all (`NO`).
* **Justification:** Early readmission prediction is a critical objective for hospital administration to reduce costs and improve care pathways. This dataset contains demographic and clinical features essential for training such a classifier.

**2. Regression:** 
* **Supported:** Yes.
* **Problem:** Predicting the length of hospital stay (`time_in_hospital`).
* **Justification:** Predicting continuous variables like the number of days a patient will stay in a bed allows hospitals to optimize resource and bed allocation.

**3. Clustering:**
* **Supported:** Yes.
* **Problem:** Unsupervised segmentation of diabetic patient profiles.
* **Justification:** Patients can be grouped based on combinations of lab procedures, age, and medication patterns to identify previously unknown high-risk subgroups.

## Final Summary Table

| Aspect | Finding |
|---|---|
| Dataset size | 101,766 rows × 50 columns |
| Numerical features | 13 |
| Categorical features | 37 |
| Missing values | Present in 7 columns (e.g., `weight`, `medical_specialty`) |
| Duplicate records | 0 duplicate rows |
| Outliers | Present (e.g., 2,557 in `num_medications`) |
| Target variable | `readmitted` (Classes: NO, >30, <30) |
| Classification | Supported |
| Regression | Supported |
| Clustering | Supported |

## Conclusion
The EDA confirmed that the Diabetes 130-US Hospitals dataset is highly appropriate for academic machine learning projects. However, the presence of significant missing values and a highly imbalanced target variable means that rigorous data preprocessing (imputation and re-sampling) will be mandatory before any machine learning models can be effectively trained.
