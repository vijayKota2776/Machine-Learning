import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Re-generate Synthetic Loan Data from Assignment 8
np.random.seed(42)
n_samples = 400

income = np.random.normal(60000, 20000, n_samples)
credit_score = np.random.normal(650, 80, n_samples)
loan_amount = np.random.normal(20000, 8000, n_samples)
years_employed = np.random.uniform(0, 15, n_samples)
debt_to_income = np.random.uniform(0.1, 0.6, n_samples)

# We define a slightly noisier logic this time so the Random Forest can show its power over a single tree
approval_score = (credit_score > 700).astype(int) * 3 + \
                 ((income > 50000) & (debt_to_income < 0.4)).astype(int) * 2 + \
                 (years_employed > 3).astype(int) * 1 + \
                 (loan_amount < 15000).astype(int) * 1 + \
                 np.random.randn(n_samples) * 0.5  # Added noise

y = (approval_score >= 3).astype(int) # 1 = Approved, 0 = Denied
X = pd.DataFrame({
    'Income': income,
    'Credit_Score': credit_score,
    'Loan_Amount': loan_amount,
    'Years_Employed': years_employed,
    'Debt_To_Income': debt_to_income
})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train Single Decision Tree (for comparison)
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)

# 3. Train Random Forest Ensemble
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# 4. Compare Performance
results = pd.DataFrame({
    'Model': ['Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, dt_preds), accuracy_score(y_test, rf_preds)],
    'Precision': [precision_score(y_test, dt_preds), precision_score(y_test, rf_preds)],
    'Recall': [recall_score(y_test, dt_preds), recall_score(y_test, rf_preds)]
})

print("Performance Comparison:")
print(results)
print("\nLaTeX Output for Report:")
print(results.to_latex(index=False, float_format="%.4f"))

# 5. Extract and Plot Feature Importances
importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Random Forest Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='indigo', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance (Gini)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('feature_importance.png', bbox_inches='tight')
print("\nSaved feature_importance.png")
