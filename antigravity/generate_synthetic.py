import numpy as np
import tifffile
import os
import cv2

def generate_synthetic_data():
    # Shape: 100x100
    # Pit 1 (20, 20): ALIVE (Pit -> Planted -> Planted/Big)
    # Pit 2 (50, 50): EMPTY (Pit -> Empty -> Empty)
    # Pit 3 (80, 80): DEAD (Pit -> Planted -> Empty)
    
    # OP1: All have pits
    # Background Light (Soil)
    img_op1 = np.full((100, 100, 3), 200, dtype=np.uint8) 
    # Dark Pits
    cv2.circle(img_op1, (20, 20), 6, (50, 50, 50), -1)
    cv2.circle(img_op1, (50, 50), 6, (50, 50, 50), -1)
    cv2.circle(img_op1, (80, 80), 6, (50, 50, 50), -1)
    
    op1_path = os.path.join("data", "cache", "test", "op1_post_pitting")
    os.makedirs(op1_path, exist_ok=True)
    tifffile.imwrite(os.path.join(op1_path, "synthetic.tif"), img_op1)
    
    # OP2: Pit 1 and 3 are planted
    # Background Light
    img_op2 = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.circle(img_op2, (20, 20), 6, (0, 255, 0), -1) # Bright Green
    # Pit 2 Empty (Just background)
    cv2.circle(img_op2, (80, 80), 6, (0, 255, 0), -1) # Bright Green
    
    op2_path = os.path.join("data", "cache", "test", "op2_post_plantating")
    os.makedirs(op2_path, exist_ok=True)
    tifffile.imwrite(os.path.join(op2_path, "synthetic.tif"), img_op2)
    
    # OP3: Only Pit 1 is Alive. Pit 3 is Dead (Empty).
    img_op3 = np.full((100, 100, 3), 200, dtype=np.uint8)
    cv2.circle(img_op3, (20, 20), 8, (0, 255, 0), -1) # Bright Green
    # Pit 3 Empty (Dead) - Just background
    
    op3_path = os.path.join("data", "cache", "test", "op3_post_sowing")
    os.makedirs(op3_path, exist_ok=True)
    tifffile.imwrite(os.path.join(op3_path, "synthetic.tif"), img_op3)
    
    print("Created synthetic OP1/OP2/OP3 images (Alive, Empty, Dead).")

if __name__ == "__main__":
    generate_synthetic_data()
