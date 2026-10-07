import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

# 1. Generate the Student Pass/Fail Dataset (From Assignment 7)
np.random.seed(42)
n_students = 200
hours_studied = np.random.uniform(0, 10, n_students)
attendance = np.random.uniform(30, 100, n_students)

logic = 5 * hours_studied + 0.5 * attendance + np.random.randn(n_students) * 10
y = (logic > 65).astype(int) # 1 = Pass, 0 = Fail
X = np.column_stack((hours_studied, attendance))

# 2. Train the Model
print("Training Logistic Regression model...")
model = LogisticRegression()
model.fit(X, y)

# 3. Save the model to disk using joblib
model_filename = 'student_classifier.pkl'
joblib.dump(model, model_filename)

print(f"Model successfully saved to '{model_filename}'")
