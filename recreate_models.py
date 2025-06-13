import os
import joblib
import numpy as np

# Ensure we're using numpy 1.24.3
print(f"Using numpy version: {np.__version__}")

# Define paths
input_dir = "model/checkpoints"
output_dir = "model/checkpoints"

# Load and re-save SVM model
svm_input_path = os.path.join(input_dir, "svm_20250524_121655.pkl")
svm_output_path = os.path.join(output_dir, "svm_model_numpy_1243.pkl")

svm_model = joblib.load(svm_input_path)
joblib.dump(svm_model, svm_output_path)
print(f"Re-saved SVM model to: {svm_output_path}")

# Load and re-save feature scaler
scaler_input_path = os.path.join(input_dir, "feature_scaler.pkl")
scaler_output_path = os.path.join(output_dir, "feature_scaler_numpy_1243.pkl")

scaler = joblib.load(scaler_input_path)
joblib.dump(scaler, scaler_output_path)
print(f"Re-saved feature scaler to: {scaler_output_path}")
