import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Dataset (Breast Cancer Dataset from Assignment 12)
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Standardize Data
# Neural Networks are extremely sensitive to unscaled data, so scaling is critical.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Baseline Model (Logistic Regression)
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict(X_test_scaled)
log_reg_acc = accuracy_score(y_test, log_reg_preds)

# 5. Train the Mini Neural Network (MLPClassifier)
# A simple architecture with one hidden layer of 100 neurons
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_preds = mlp.predict(X_test_scaled)
mlp_acc = accuracy_score(y_test, mlp_preds)

# 6. Report Comparison
results = pd.DataFrame({
    'Model': ['Logistic Regression (Baseline)', 'Neural Network (MLP)'],
    'Accuracy': [log_reg_acc, mlp_acc]
})

print("Model Comparison Results:")
print(results)
print("\nLaTeX Output:")
print(results.to_latex(index=False, float_format="%.4f"))

# Plot accuracy comparison
plt.figure(figsize=(8, 5))
bars = plt.bar(results['Model'], results['Accuracy'], color=['skyblue', 'salmon'])
plt.ylim(0.9, 1.0)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison: Logistic Regression vs Neural Network')

# Add text labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom')

plt.savefig('comparison.png')
print("\nSaved comparison.png")
