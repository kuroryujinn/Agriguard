import os
import tifffile
import numpy as np
from PIL import Image

def generate_preview(tif_path, size=(512, 512)):
    """Generates a JPEG preview from a .tif file using tifffile."""
    preview_dir = os.path.dirname(tif_path).replace('cache', 'cache/previews') # Hacky but efficient for now
    # Ensure preview directory structure mirrors cache or just put them in data/cache/previews flatly?
    # Requirement: "data/cache/previews/"
    
    # Better path handling
    base_name = os.path.basename(tif_path)
    preview_path = os.path.join("data", "cache", "previews", f"{base_name}.jpg")
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)

    if os.path.exists(preview_path):
        return preview_path

    try:
        # Read the image
        img = tifffile.imread(tif_path)
        
        # Handle different shapes/channels
        if img.ndim == 3:
            # Assuming (H, W, C) or (C, H, W) - rasterio vs tifffile nuances
            # TiFFfile usually returns (Page, H, W) or (H, W, C)
            if img.shape[0] < 10: # Likely (C, H, W) e.g., (3, 2000, 2000) or (4, 2000, 2000)
                img = np.transpose(img, (1, 2, 0))
            
            # Select RGB only if more channels
            if img.shape[2] >= 3:
                img = img[:, :, :3]
        
        # Normalize if needed (uint16 -> uint8)
        if img.dtype == 'uint16':
            img = (img / 256).astype('uint8')
        elif img.dtype == 'float32':
             img = (img * 255).astype('uint8') # Simplistic normalization
        
        # Convert to PIL for resizing
        pil_img = Image.fromarray(img)
        pil_img.thumbnail(size)
        
        # Convert to RGB (jpeg doesn't support alpha)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        pil_img.save(preview_path, "JPEG", quality=80)
        return preview_path

    except Exception as e:
        print(f"Error generating preview for {tif_path}: {e}")
        return None
