import cv2
import numpy as np
import os
import random

# Configuration
OUTPUT_DIR = "dataset_cv2"
IMG_SIZE = 32
NUM_SAMPLES_PER_SYMBOL = 25
SYMBOLS = ['+', '-', 'x', 'div'] 

# Create output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def create_image_cv2(symbol, filename):
    # 1. Create a white background (255) 32x32 image
    # We use uint8 because image pixels are 0-255
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
    
    # Calculate center
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    
    # Add random shift (noise)
    shift_x = random.randint(-2, 2)
    shift_y = random.randint(-2, 2)
    
    # Drawing settings
    color = 0  # Black ink
    thickness = 2
    ext = 10 # length of lines from center

    if symbol == '+':
        # Horizontal
        cv2.line(img, (cx - ext + shift_x, cy + shift_y), (cx + ext + shift_x, cy + shift_y), color, thickness)
        # Vertical
        cv2.line(img, (cx + shift_x, cy - ext + shift_y), (cx + shift_x, cy + ext + shift_y), color, thickness)
        
    elif symbol == '-':
        # Horizontal only
        cv2.line(img, (cx - ext + shift_x, cy + shift_y), (cx + ext + shift_x, cy + shift_y), color, thickness)
        
    elif symbol == 'x': # Multiplication
        # Diagonal 1
        cv2.line(img, (cx - ext + shift_x, cy - ext + shift_y), (cx + ext + shift_x, cy + ext + shift_y), color, thickness)
        # Diagonal 2
        cv2.line(img, (cx + ext + shift_x, cy - ext + shift_y), (cx - ext + shift_x, cy + ext + shift_y), color, thickness)
        
    elif symbol == 'div': # Division (Obelus ÷)
        # Horizontal bar
        cv2.line(img, (cx - ext + shift_x, cy + shift_y), (cx + ext + shift_x, cy + shift_y), color, thickness)
        # Top dot (filled circle, radius 1)
        cv2.circle(img, (cx + shift_x, cy - 5 + shift_y), 1, color, -1)
        # Bottom dot
        cv2.circle(img, (cx + shift_x, cy + 5 + shift_y), 1, color, -1)

    # Save the image
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)

print("Generating images with OpenCV...")

for symbol in SYMBOLS:
    for i in range(NUM_SAMPLES_PER_SYMBOL):
        filename = f"{symbol}_{i}.png"
        create_image_cv2(symbol, filename)

print(f"Done! 100 images saved in '{OUTPUT_DIR}' folder.")

