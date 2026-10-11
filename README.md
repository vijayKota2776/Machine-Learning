# Machine Learning Portfolio & Assignments

A structured collection of 15 Machine Learning assignments covering fundamental to advanced machine learning concepts, data preparation, supervised/unsupervised algorithms, evaluation methodologies, neural networks, and model serialization/deployment pipelines.

**Author:** Vijay Kota  
**Student ID:** 4074  
**Repository:** [vijayKota2776/Machine-Learning](https://github.com/vijayKota2776/Machine-Learning)

---

## 📚 Overview

This repository demonstrates hands-on implementation of core machine learning workflows using Python and industry-standard libraries (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, and `joblib`). Each assignment contains documented code, generated visual artifacts (plots, charts, decision trees), and comprehensive academic PDF reports formatted per course standards.

---

## 🗂️ Table of Contents

| # | Directory | Topic | Key Techniques & Models | Deliverables |
|---|---|---|---|---|
| **01** | [`Assignment1_Applied_Reasoning`](./Assignment1_Applied_Reasoning) | Applied Reasoning | ML intuition, ethics, problem framing | `vijaykota_4074_ass1.pdf` |
| **02** | [`Assignment2_Exploratory_Data_Analysis`](./Assignment2_Exploratory_Data_Analysis) | Exploratory Data Analysis (EDA) | Data distribution, outlier detection, correlations | Jupyter Notebooks, `vijaykota_4074_ass2.pdf`, `EDA_Report.md` |
| **03** | [`Assignment3_ML_Toolkit`](./Assignment3_ML_Toolkit) | ML Toolkit | Vectorized NumPy computations, Pandas operations | `main.py`, `ml_utils.py`, `vijaykota_4074_ass3.pdf` |
| **04** | [`Assignment4_Applied_Data_Preparation`](./Assignment4_Applied_Data_Preparation) | Applied Data Preparation | Imputation, One-Hot/Label encoding, Feature scaling | `data_preparation.py`, `explore.py`, `vijaykota_4074_ass4.pdf` |
| **05** | [`Assignment5_Regression_Project`](./Assignment5_Regression_Project) | Regression Project | Ordinary Least Squares, MSE, RMSE, R² metrics | `Regression_Mini_Project.ipynb`, report, plots |
| **06** | [`Assignment6_Polynomial_Regression_Challenge`](./Assignment6_Polynomial_Regression_Challenge) | Polynomial Regression Challenge | Degree comparison (1, 2, 5), Bias-Variance tradeoff, Overfitting | `polynomial_regression.py`, `polynomial_comparison.png`, `vijaykota_4074_ass6.pdf` |
| **07** | [`Assignment7_Logistic_Regression`](./Assignment7_Logistic_Regression) | Logistic Regression | Sigmoid activation, Decision thresholds, Confusion Matrix | `logistic_regression.py`, `confusion_matrix.png`, `vijaykota_4074_ass7.pdf` |
| **08** | [`Assignment8_Decision_Tree`](./Assignment8_Decision_Tree) | Decision Trees | Gini Impurity, Tree pruning (`max_depth`), Tree visualization | `decision_tree.py`, `decision_tree.png`, `vijaykota_4074_ass8.pdf` |
| **09** | [`Assignment9_Model_Evaluation_Showdown`](./Assignment9_Model_Evaluation_Showdown) | Model Evaluation Showdown | Logistic Regression vs. KNN vs. Decision Tree multi-metric comparison | `model_evaluation.py`, `confusion_matrices.png`, `vijaykota_4074_ass9.pdf` |
| **10** | [`Assignment10_Cross_Validation`](./Assignment10_Cross_Validation) | Cross-Validation Robustness Test | 10-Fold Stratified K-Fold CV, Variance analysis (Mean & Std Dev) | `cross_validation.py`, `vijaykota_4074_ass10.pdf` |
| **11** | [`Assignment11_Customer_Segmentation`](./Assignment11_Customer_Segmentation) | Customer Segmentation | K-Means Clustering, Elbow Method (Inertia), Business Profiling | `kmeans_clustering.py`, `elbow_method.png`, `clusters.png`, `vijaykota_4074_ass11.pdf` |
| **12** | [`Assignment12_PCA_Visualization`](./Assignment12_PCA_Visualization) | PCA Visualization Explorer | Dimensionality Reduction to 2D, Explained Variance Ratio | `pca_explorer.py`, `pca_visualization.png`, `vijaykota_4074_ass12.pdf` |
| **13** | [`Assignment13_Random_Forest`](./Assignment13_Random_Forest) | Random Forest Ensemble Boost | Bagging, Forest vs. Single Tree comparison, Gini Feature Importance | `random_forest.py`, `feature_importance.png`, `vijaykota_4074_ass13.pdf` |
| **14** | [`Assignment14_Mini_Neural_Network`](./Assignment14_Mini_Neural_Network) | Mini Neural Network Classifier | Multi-Layer Perceptron (`MLPClassifier`), ReLU, Adam, baseline comparison | `neural_network.py`, `comparison.png`, `vijaykota_4074_ass14.pdf` |
| **15** | [`Assignment15_Model_Deployment_Pipeline`](./Assignment15_Model_Deployment_Pipeline) | Model Deployment Pipeline | Model persistence (`joblib`), production inference script, error handling | `train_and_save.py`, `load_and_predict.py`, `student_classifier.pkl`, `vijaykota_4074_ass15.pdf` |

---

## 🔍 Detailed Assignment Breakdown

### Assignment 1: Applied Reasoning
- Foundational exploration of machine learning problem framing, ethical boundaries, and problem categorization.
- Formal report: `vijaykota_4074_ass1.pdf`.

### Assignment 2: Exploratory Data Analysis (EDA)
- Comprehensive univariate and bivariate statistical analysis on tabular datasets.
- Handled distributions, skewness, outliers, and visual correlation heatmaps.

### Assignment 3: ML Toolkit
- Vectorized array operations, matrix mathematics with NumPy, and clean DataFrame manipulations with Pandas.
- Core utilities implemented in `ml_utils.py` and demonstrated via `main.py`.

### Assignment 4: Applied Data Preparation
- Complete end-to-end data preprocessing pipeline: missing value imputation strategies, categorical encoding (One-Hot & Label), numerical standard scaling, and train-test splitting.

### Assignment 5: Regression Project
- Linear regression modeling for predicting student academic performance.
- Evaluation utilizing Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).

### Assignment 6: Polynomial Regression Challenge
- Tested Polynomial features across degrees 1 (linear baseline), 2 (optimal fit), and 5 (severe overfitting).
- Analyzed the Bias-Variance tradeoff and verified performance with train vs. test MSE tracking.

### Assignment 7: Logistic Regression
- Binary classification of student pass/fail outcomes using standard Logistic Regression.
- Analyzed decision boundaries, odds ratios, probability thresholds, and evaluated with Confusion Matrix and classification reports.

### Assignment 8: Decision Tree Classifier
- Trained a Decision Tree classifier using Gini Impurity splits.
- Investigated depth constraints (`max_depth=3`) to prevent overfitting and generated high-resolution tree visualization diagrams.

### Assignment 9: Model Evaluation Showdown
- Head-to-head comparison of Logistic Regression, K-Nearest Neighbors (KNN), and Decision Trees on identical splits.
- Compared Accuracy, Precision, Recall, F1-Score, and comparative Confusion Matrix subplots with concrete deployment trade-off justifications.

### Assignment 10: Cross-Validation Robustness Test
- Evaluated model variance across 10 folds using `StratifiedKFold`.
- Detailed per-fold accuracy distributions, standard deviation interpretation, and diagnosed split sensitivity.

### Assignment 11: Customer Segmentation (K-Means)
- Unsupervised clustering applied to annual income and spending score metrics.
- Utilized the Elbow Method (Inertia curve) to locate optimal $k=5$ and generated actionable business personas for targeted marketing.

### Assignment 12: PCA Visualization Explorer
- Dimensionality reduction projecting multi-dimensional feature space onto 2 principal components.
- Calculated explained variance ratios and produced 2D scatter plots demonstrating class separability.

### Assignment 13: Random Forest Ensemble Boost
- Ensemble bagging with 100 decision trees to evaluate performance gains over a standalone tree.
- Extracted and visualized Gini feature importances to reveal critical predictive signals.

### Assignment 14: Mini Neural Network Classifier
- Configured a Multi-Layer Perceptron (`MLPClassifier`) with hidden layer architectures, ReLU activations, and Adam optimization.
- Benchmarked against linear baselines and critiqued trade-offs between representation capacity and interpretability.

### Assignment 15: Model Deployment Pipeline
- Decoupled training and deployment into independent modular scripts.
- Serialized pipeline and metadata using `joblib`.
- Created robust inference script (`load_and_predict.py`) with explicit file-existence validation (`FileNotFoundError`), schema checks, and batch predictions on unseen data.

---

## 🛠️ Environment Setup & Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/vijayKota2776/Machine-Learning.git
cd Machine-Learning
```

### Recommended Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

---

## 🚀 Running Any Assignment

Each assignment directory is self-contained. Navigate to any assignment folder and execute the corresponding Python script or Jupyter notebook:

```bash
# Example: Run Assignment 15 Deployment Pipeline
cd Assignment15_Model_Deployment_Pipeline
python3 train_and_save.py    # Trains and serializes the model to .pkl
python3 load_and_predict.py   # Loads model and runs inference with error handling

# Example: Run Assignment 11 Customer Segmentation
cd ../Assignment11_Customer_Segmentation
python3 kmeans_clustering.py  # Generates elbow curve and cluster plots
```

---

## 📄 License & Attribution

This repository is maintained by Vijay Kota for academic coursework and portfolio reference.  
Feel free to reference or adapt the code for learning purposes.
