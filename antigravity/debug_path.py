import os
import json

cache_dir = os.path.join("data", "cache")
field = "test"
stage = "op2_post_plantating"
name = "synthetic.tif"

target_dir = os.path.join(cache_dir, f"{field}/{stage}")
file_path = os.path.join(target_dir, name)

print(f"Checking: {file_path}")
print(f"Exists: {os.path.exists(file_path)}")
print(f"Abs: {os.path.abspath(file_path)}")

# Check manifest loading
manifest = "data/manifests/test.sample.json"
with open(manifest) as f:
    data = json.load(f)
    print("Manifest data:")
    for d in data:
        if d['stage'] == stage:
            print(d)
