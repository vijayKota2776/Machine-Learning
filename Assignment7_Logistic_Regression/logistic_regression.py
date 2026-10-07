import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Generate synthetic student dataset
# Features: Hours Studied (0-10), Attendance Percentage (0-100)
np.random.seed(42)
n_students = 200
hours_studied = np.random.uniform(0, 10, n_students)
attendance = np.random.uniform(30, 100, n_students)

# We define a hidden logic: Pass if 5 * hours + 0.5 * attendance > 65 + some noise
logic = 5 * hours_studied + 0.5 * attendance + np.random.randn(n_students) * 10
y = (logic > 65).astype(int) # 1 = Pass, 0 = Fail

X = np.column_stack((hours_studied, attendance))

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")

# Generate Confusion Matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fail', 'Pass'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Student Pass/Fail Classifier')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

# 5. Predict on 3 new records
# Record A: High hours, high attendance
# Record B: Low hours, low attendance
# Record C: Moderate hours, low attendance
new_records = np.array([
    [8.5, 95.0], # Student A
    [2.0, 40.0], # Student B
    [6.0, 50.0]  # Student C
])

predictions = model.predict(new_records)
probabilities = model.predict_proba(new_records)[:, 1] # Probability of passing

print("\nPredictions on New Records:")
for i, (record, pred, prob) in enumerate(zip(new_records, predictions, probabilities)):
    status = "Pass" if pred == 1 else "Fail"
    print(f"Student {chr(65+i)} (Hours: {record[0]}, Attendance: {record[1]}%): {status} (Probability: {prob:.2f})")
