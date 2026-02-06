import random
import os
import shutil

# Configuration
IMAGE_SIZE = 40
IMAGES_PER_CLASS = 250
symbols = ["plus", "minus", "multiply", "divide"]

# Setup Directories
if os.path.exists("data"):
    shutil.rmtree("data")
for sym in symbols:
    os.makedirs(f"data/{sym}", exist_ok=True)

# Helper Functions
def blank_image():
    return [0] * (IMAGE_SIZE * IMAGE_SIZE)

def add_noise(img, level=50):
    for i in range(len(img)):
        if random.randint(0, 100) < 15:
            img[i] = max(img[i], random.randint(0, level))
    return img

def draw_plus(img):
    mid = IMAGE_SIZE // 2
    for i in range(IMAGE_SIZE):
        img[mid * IMAGE_SIZE + i] = 255      # Horizontal line
        img[i * IMAGE_SIZE + mid] = 255      # Vertical line
    return img

def draw_minus(img):
    mid = IMAGE_SIZE // 2
    for i in range(IMAGE_SIZE):
        img[mid * IMAGE_SIZE + i] = 255      # Horizontal line only
    return img

def draw_multiply(img):
    for i in range(IMAGE_SIZE):
        img[i * IMAGE_SIZE + i] = 255        # Main diagonal
        img[i * IMAGE_SIZE + (IMAGE_SIZE - i - 1)] = 255 # Anti-diagonal
    return img

def draw_divide(img):
    for i in range(IMAGE_SIZE):
        img[i * IMAGE_SIZE + (IMAGE_SIZE - i - 1)] = 255 # Anti-diagonal only
    return img

# Generate Dataset
print("Generating dataset files...")
for sym in symbols:
    for idx in range(1, IMAGES_PER_CLASS + 1):
        img = blank_image()
        
        if sym == "plus":      draw_plus(img)
        elif sym == "minus":   draw_minus(img)
        elif sym == "multiply": draw_multiply(img)
        elif sym == "divide":  draw_divide(img)
        
        add_noise(img)
        
        with open(f"data/{sym}/{sym}{idx}.txt", "w") as f:
            f.write(" ".join(map(str, img)))

print(f"Success! Generated {IMAGES_PER_CLASS * 4} files.")
print("Dataset generation complete.\n")

print("-- SCROLL DOWN TO SEE ALL 4 SYMBOLS ---")

def show_matrix(filepath, title):
    if not os.path.exists(filepath): return
    print(f"\n{title} ({filepath})")
    print("=" * 40)
    
    with open(filepath, 'r') as f:
        content = list(map(int, f.read().split()))
    
    # Print 40x40 grid
    for r in range(IMAGE_SIZE):
        row_vals = content[r*IMAGE_SIZE : (r+1)*IMAGE_SIZE]
        # formatting specific to match report visualization style
        print(" ".join(f"{val:3}" for val in row_vals))
    
    print("=" * 40 + "\n\n")

# Show examples
show_matrix("data/plus/plus1.txt", "MATRIX VIEW: PLUS (+)")
show_matrix("data/minus/minus1.txt", "MATRIX VIEW: MINUS (-)")
show_matrix("data/multiply/multiply1.txt", "MATRIX VIEW: MULTIPLY (*)")
show_matrix("data/divide/divide1.txt", "MATRIX VIEW: DIVIDE (/)")
