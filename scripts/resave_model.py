import os
import joblib
import numpy as np
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Joblib version: {joblib.__version__}")

# Define paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model" / "checkpoints"

# Input files
OLD_MODEL = MODEL_DIR / "svm_20250524_121655.pkl"
OLD_SCALER = MODEL_DIR / "feature_scaler.pkl"

# Output files
NEW_MODEL = MODEL_DIR / "svm_np1243.pkl"
NEW_SCALER = MODEL_DIR / "scaler_np1243.pkl"

def main():
    print("\n=== Re-saving model with current NumPy version ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model directory: {MODEL_DIR}")
    
    # Check if input files exist
    if not OLD_MODEL.exists():
        print(f"Error: Model file not found at {OLD_MODEL}")
        return
    if not OLD_SCALER.exists():
        print(f"Error: Scaler file not found at {OLD_SCALER}")
        return
    
    try:
        # Load the old model and scaler
        print("\nLoading original model...")
        model = joblib.load(OLD_MODEL)
        print("Model loaded successfully")
        
        print("\nLoading original scaler...")
        scaler = joblib.load(OLD_SCALER)
        print("Scaler loaded successfully")
        
        # Save with the current NumPy version
        print(f"\nSaving model to {NEW_MODEL}")
        joblib.dump(model, NEW_MODEL)
        
        print(f"Saving scaler to {NEW_SCALER}")
        joblib.dump(scaler, NEW_SCALER)
        
        # Verify the saved files
        print("\nVerifying saved files...")
        _ = joblib.load(NEW_MODEL)
        _ = joblib.load(NEW_SCALER)
        
        print("\n✅ Success! Model and scaler re-saved with current NumPy version.")
        print(f"New model: {NEW_MODEL}")
        print(f"New scaler: {NEW_SCALER}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    main()
