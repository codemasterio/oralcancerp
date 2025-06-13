import os
import sys
import requests
import urllib.request
import time
from pathlib import Path

def download_file(url, destination):
    """
    Download a file from a URL to the specified destination.
    Handles Google Drive large file downloads correctly.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        print(f"Downloading {os.path.basename(destination)}...")
        
        # For Google Drive files
        if 'drive.google.com' in url:
            # Extract the file ID from the URL
            file_id = url.split('id=')[-1].split('&')[0]
            direct_download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
            
            # Use requests with a session to handle the download
            session = requests.Session()
            
            # First request to get the confirmation token
            response = session.get(direct_download_url, stream=True)
            token = None
            
            # Save the file
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(32768):
                    if chunk:
                        f.write(chunk)
            
            # Verify file was downloaded
            if os.path.exists(destination) and os.path.getsize(destination) > 0:
                print(f"Successfully downloaded {os.path.basename(destination)}")
                return True
            else:
                raise Exception("Downloaded file is empty or failed")
        
        # For direct downloads
        else:
            with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully downloaded {os.path.basename(destination)}")
            return True
            
    except Exception as e:
        print(f"Error downloading {os.path.basename(destination)}: {str(e)}")
        # Clean up partially downloaded file if it exists
        if os.path.exists(destination):
            os.remove(destination)
        return False
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """
    Main function to download all required model files.
    Replace the Google Drive share links with your actual file links.
    """
    # Create model directory if it doesn't exist
    model_dir = Path("model/checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary of model files to download
    # Replace these with your actual Google Drive share links
    model_files = {
        # Format: "Google_Drive_Share_Link": model_dir / "filename.pkl",
        "https://drive.google.com/file/d/1nQSk8_DNoHuAjRhw56XsM8h9IY_nHYSE/view?usp=drive_link": model_dir / "latest_svm_model.pkl",
        "https://drive.google.com/file/d/1G21yXs-KWFVOr0GDFb-_NbuyZNvjrcaf/view?usp=sharing": model_dir / "feature_scaler.pkl",
        "https://drive.google.com/file/d/1yNai3RxxOFWNN57-p-MAbPOGRWJom-8l/view?usp=sharing": model_dir / "svm_20250524_121655_metadata.pkl"
    }
    
    print("Starting model file download...\n")
    
    # Download each file
    results = []
    for url, dest in model_files.items():
        if not url.startswith("https://drive.google.com"):
            print(f"Error: Please update the download link for {dest.name}")
            results.append(False)
            continue
            
        print(f"\n--- Downloading {dest.name} ---")
        success = download_file(url, str(dest))
        results.append(success)
        
        # Add a small delay between downloads
        time.sleep(1)
    
    # Print summary
    print("\n" + "="*50)
    if all(results):
        print("✅ All model files downloaded successfully!")
    else:
        failed = [f for f, s in zip(model_files.values(), results) if not s]
        print(f"❌ Some files failed to download: {', '.join(str(f.name) for f in failed)}")
        print("Please check the error messages above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
