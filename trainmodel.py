import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
X = []
y = []

with open("data/asl_data.csv", "r") as file:
    reader = csv.reader(file)

    for row in reader:
        if len(row) == 64:
            landmarks = [float(value) for value in row[:63]]
            label = row[63]

            X.append(landmarks)
            y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Labels:", y)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "asl_model.pkl")

print("Model trained successfully!")
print("Model saved as asl_model.pkl")