import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

# 1. Generate Synthetic Loan Data (Reused from Assignment 8)
np.random.seed(42)
n_samples = 400

income = np.random.normal(60000, 20000, n_samples)
credit_score = np.random.normal(650, 80, n_samples)
loan_amount = np.random.normal(20000, 8000, n_samples)
years_employed = np.random.uniform(0, 15, n_samples)
debt_to_income = np.random.uniform(0.1, 0.6, n_samples)

approval_score = (credit_score > 700).astype(int) * 3 + \
                 ((income > 50000) & (debt_to_income < 0.4)).astype(int) * 2 + \
                 (years_employed > 3).astype(int) * 1 + \
                 (loan_amount < 15000).astype(int) * 1

y = (approval_score >= 3).astype(int) # Target
X = pd.DataFrame({
    'Income': income,
    'Credit_Score': credit_score,
    'Loan_Amount': loan_amount,
    'Years_Employed': years_employed,
    'Debt_To_Income': debt_to_income
})

# 2. Setup Classifier and K-Fold Cross-Validation
# Reusing the Decision Tree from earlier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 3. Perform Cross-Validation
fold_accuracies = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# 4. Report Results
print("K-Fold Cross-Validation Results (k=10)\n")

fold_data = []
for idx, acc in enumerate(fold_accuracies):
    fold_data.append({'Fold': idx + 1, 'Accuracy': acc})
    print(f"Fold {idx + 1}: {acc:.4f}")

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)

print(f"\nMean Accuracy: {mean_acc:.4f}")
print(f"Standard Deviation: {std_acc:.4f}")

# Optional: Output LaTeX table format
results_df = pd.DataFrame(fold_data)
print("\nLaTeX Table:")
print(results_df.to_latex(index=False, float_format="%.4f"))
