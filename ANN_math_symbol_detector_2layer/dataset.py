import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 40  
NUM_TRAIN = 200  
NUM_TEST = 25    
def generate_image(symbol, size=IMAGE_SIZE):
    """Generates a grayscale image of a symbol with random noise."""
    img = np.zeros((size, size))

    noise_level = 0.1
    img += np.random.rand(size, size) * noise_level

    margin = 8
    mid = size // 2

    if symbol == '+':
        img[margin:size-margin, mid-2:mid+2] = 1.0 
        img[mid-2:mid+2, margin:size-margin] = 1.0 
    elif symbol == '-':
        img[mid-2:mid+2, margin:size-margin] = 1.0 
    elif symbol == '*':
    
        for i in range(margin, size-margin):
            img[i, i] = 1.0
            if i+1 < size: img[i, i+1] = 1.0
        
        for i in range(margin, size-margin):
            img[i, size-1-i] = 1.0
            if size-1-i+1 < size: img[i, size-1-i+1] = 1.0
        
        img[margin:size-margin, mid-1:mid+1] = 1.0 
        img[mid-1:mid+1, margin:size-margin] = 1.0
    elif symbol == '/':
    
        for i in range(margin, size-margin):
            img[size-1-i, i] = 1.0
            if i+1 < size: img[size-1-i, i+1] = 1.0

    return np.clip(img, 0, 1)

def create_dataset(samples_per_class):
    inputs = []
    targets = []
    symbols = ['+', '-', '*', '/']
    
    for i, sym in enumerate(symbols):
        for _ in range(samples_per_class):
            img = generate_image(sym)
            flat_img = img.flatten() 
            inputs.append(flat_img)

            t = np.zeros(4)
            t[i] = 1
            targets.append(t)
            
    return np.array(inputs), np.array(targets), symbols

print("Generating Training Data...")
X_train, y_train, classes = create_dataset(NUM_TRAIN)

print("Generating Testing Data...")
X_test, y_test, _ = create_dataset(NUM_TEST)

indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print(f"Data Prepared. Input Vector Size: {X_train.shape[1]}")

print("\nSample Images (One of each symbol):")
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
sample_symbols = ['+', '-', '*', '/']

for i, ax in enumerate(axes):

    sample = generate_image(sample_symbols[i])
    ax.imshow(sample, cmap='gray')
    ax.set_title(sample_symbols[i])
    ax.axis('off')
plt.show()