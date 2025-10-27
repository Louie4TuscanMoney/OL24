"""
MAMBA MODEL AUTO-DOWNLOADER

Purpose: Download Mamba model from Google Drive on Railway startup
Author: Ontologic XYZ
Date: October 27, 2025
"""

import os
import requests
from pathlib import Path


def download_file_from_google_drive(file_id, destination):
    """
    Download a file from Google Drive
    
    Args:
        file_id: Google Drive file ID (from shareable link)
        destination: Local file path to save
    """
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)


def get_confirm_token(response):
    """Get download confirmation token from Google Drive"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    """Save streaming response to file"""
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_mamba_model(output_path: str = "MAMBA_MENTALITY_SYSTEM.pkl") -> bool:
    """
    Download Mamba model if not present
    
    Args:
        output_path: Where to save the model (default: current directory)
                    For Railway: use "/tmp/MAMBA_MENTALITY_SYSTEM.pkl"
    
    Returns:
        True if successful, False otherwise
    
    To use:
    1. Upload MAMBA_MENTALITY_SYSTEM.pkl to Google Drive
    2. Get shareable link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    3. Extract FILE_ID and replace below
    4. Add to Railway startup command
    """
    
    # Google Drive file ID from shareable link
    GOOGLE_DRIVE_FILE_ID = "1gGRfh-07VjfD--VjftmUq-G2UtxI1T-7"  # MAMBA_MENTALITY_SYSTEM.pkl
    
    # Check if model already exists
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Mamba model already exists at {output_path} ({file_size_mb:.1f} MB)")
        return True
    
    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created directory: {output_dir}")
    
    # Download model using gdown (handles large files properly!)
    print(f"📦 Downloading Mamba model from Google Drive...")
    print(f"   File ID: {GOOGLE_DRIVE_FILE_ID}")
    print(f"   Output path: {output_path}")
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, output_path, quiet=False)
        
        # Verify download
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Model downloaded successfully! ({file_size_mb:.1f} MB)")
            return True
        else:
            print(f"❌ Model download failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print(f"⚠️ System will run with synthetic predictions")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    download_mamba_model()

