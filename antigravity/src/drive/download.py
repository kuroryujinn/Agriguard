import io
import os
import json
import time
from googleapiclient.http import MediaIoBaseDownload
from src.drive.auth import get_service

CACHE_DIR = os.path.join("data", "cache")

def download_file(file_id, file_name, dest_subfolder=None, meta=None, force=False):
    """
    Downloads a file from Drive if not already cached.
    dest_subfolder: e.g., 'benkanmura/op1_post_pitting'
    meta: dict to save as .meta.json
    """
    
    if dest_subfolder:
        target_dir = os.path.join(CACHE_DIR, dest_subfolder)
    else:
        target_dir = CACHE_DIR
        
    target_dir = os.path.normpath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, file_name)
    file_path = os.path.normpath(file_path)

    # Check sidecar meta to be robust? For now just check file existence.
    if os.path.exists(file_path):
        # Update meta even if file exists?
        if meta:
            _write_meta(file_path, meta, file_id)
        return file_path

    service = get_service()
    request = service.files().get_media(fileId=file_id)
    
    print(f"Downloading {file_name} -> {target_dir}...")
    with io.FileIO(file_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
    
    if meta:
        _write_meta(file_path, meta, file_id)
    
    return file_path

def _write_meta(file_path, meta, file_id):
    meta_path = file_path + ".meta.json"
    data = {
        "file_id": file_id,
        "downloaded_at": time.time(),
        "original_meta": meta
    }
    with open(meta_path, 'w') as f:
        json.dump(data, f, indent=2)
