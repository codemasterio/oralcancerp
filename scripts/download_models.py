import os
import sys
import requests
import urllib.request
from pathlib import Path

def download_file(url, destination):
    """Download a file from a URL to the specified destination."""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        print(f"Downloading {url} to {destination}...")
        
        # Download the file
        with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
            
        print(f"Successfully downloaded {os.path.basename(destination)}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create model directory if it doesn't exist
    model_dir = Path("model/checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # List of model files to download (replace with your actual model URLs)
    model_files = {
        "https://example.com/path/to/latest_svm_model.pkl": model_dir / "latest_svm_model.pkl",
        "https://example.com/path/to/feature_scaler.pkl": model_dir / "feature_scaler.pkl",
    }
    
    # Download each file
    success = all(download_file(url, str(dest)) for url, dest in model_files.items())
    
    if success:
        print("\nAll model files downloaded successfully!")
    else:
        print("\nSome files failed to download. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
