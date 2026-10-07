import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Loan Data
np.random.seed(42)
n_samples = 300

# Features
income = np.random.normal(60000, 20000, n_samples)
credit_score = np.random.normal(650, 80, n_samples)
loan_amount = np.random.normal(20000, 8000, n_samples)
years_employed = np.random.uniform(0, 15, n_samples)
debt_to_income = np.random.uniform(0.1, 0.6, n_samples)

# Logic for Approval:
# High credit score OR (Good income AND low DTI) generally pass
approval_score = (credit_score > 700).astype(int) * 3 + \
                 ((income > 50000) & (debt_to_income < 0.4)).astype(int) * 2 + \
                 (years_employed > 3).astype(int) * 1 + \
                 (loan_amount < 15000).astype(int) * 1

y = (approval_score >= 3).astype(int) # 1 = Approved, 0 = Denied

X = pd.DataFrame({
    'Income': income,
    'Credit_Score': credit_score,
    'Loan_Amount': loan_amount,
    'Years_Employed': years_employed,
    'Debt_To_Income': debt_to_income
})

# 2. Split Data and Train Decision Tree
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using max_depth=3 to keep the visualization interpretable
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 3. Visualize the Decision Tree
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=X.columns, class_names=['Denied', 'Approved'], filled=True, rounded=True, proportion=True)
plt.title('Loan Approval Decision Tree')
plt.savefig('decision_tree.png', bbox_inches='tight')
print("Saved decision_tree.png")

# 4. Identify Top 3 Features
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

print("\nTop 3 Features:")
print(feature_importance_df.head(3))
