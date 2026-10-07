import sys
import numpy as np
import joblib

model_filename = 'student_classifier.pkl'

# 1. Attempt to load the model with Error Handling
try:
    print(f"Attempting to load model from '{model_filename}'...")
    model = joblib.load(model_filename)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"\n[ERROR] The model file '{model_filename}' was not found!")
    print("Please ensure you run 'train_and_save.py' first to generate the model file.")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred while loading the model: {e}")
    sys.exit(1)

# 2. Define 3 new unseen student records
# Feature 1: Hours Studied, Feature 2: Attendance Percentage
new_students = np.array([
    [9.0, 95.0],  # Student 1: High hours, high attendance
    [1.5, 45.0],  # Student 2: Low hours, low attendance
    [5.0, 60.0]   # Student 3: Moderate hours, moderate attendance
])

# 3. Predict without retraining
predictions = model.predict(new_students)
probabilities = model.predict_proba(new_students)[:, 1]

print("\n--- Predictions on New Data ---")
for i, (record, pred, prob) in enumerate(zip(new_students, predictions, probabilities)):
    status = "Pass" if pred == 1 else "Fail"
    print(f"Student {i+1} (Hours: {record[0]}, Attendance: {record[1]}%): {status} (Confidence: {prob*100:.1f}%)")
