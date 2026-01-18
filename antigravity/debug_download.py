import os
import json
from src.drive.download import download_file

def test_download():
    field = "test"
    item = {"id":"syn2","name":"synthetic.tif","stage":"op2_post_plantating","sampled":True}
    subfolder = f"{field}/{item['stage']}"
    
    print(f"Subfolder: {subfolder}")
    print(f"CACHE_DIR (imported): {os.path.join('data', 'cache')}")
    
    try:
        path = download_file(item['id'], item['name'], dest_subfolder=subfolder, meta=item)
        print(f"Success! Path: {path}")
    except Exception as e:
        print(f"Failure: {e}")

if __name__ == "__main__":
    test_download()
