import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION ---
DATA_DIR = "dataset_cv2"
IMG_SIZE = 32
INPUT_NODES = (IMG_SIZE * IMG_SIZE) + 1  
OUTPUT_NODES = 4   

# --- THE SWEET SPOT ---
# Increased Learning Rate significantly (faster learning)
LEARNING_RATE = 0.05  
# Decreased Epochs (since we learn faster, we don't need as many loops)
EPOCHS = 500
# ----------------------

LABEL_MAP = {'+': 0, '-': 1, 'x': 2, 'div': 3}
REVERSE_MAP = {0: '+', 1: '-', 2: 'x', 3: 'div'}

def sigmoid(x):
    # Using a slightly safer sigmoid to prevent overflow
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def load_data():
    print("Loading images...")
    data = []
    labels = []
    image_paths = glob.glob(os.path.join(DATA_DIR, "*.png"))
    
    if not image_paths:
        print("Error: No images found!")
        exit()

    for path in image_paths:
        filename = os.path.basename(path)
        symbol = filename.split('_')[0]
        img = cv2.imread(path, 0)
        
        flattened = img.reshape(-1) / 255.0
        flattened_with_bias = np.append(flattened, 1.0)
        
        data.append(flattened_with_bias)
        labels.append(LABEL_MAP[symbol])
        
    return np.array(data), np.array(labels)

inputs, targets = load_data()
n_samples = inputs.shape[0]

# Initialize Weights
weights = np.random.uniform(-0.1, 0.1, (INPUT_NODES, OUTPUT_NODES))

print(f"Training on {n_samples} images for {EPOCHS} epochs (LR={LEARNING_RATE})...")

for epoch in range(EPOCHS):
    total_error = 0
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for i in indices:
        x = inputs[i].reshape(1, INPUT_NODES)
        
        desired = np.zeros((1, OUTPUT_NODES))
        desired[0, targets[i]] = 1
        
        # Forward
        weighted_sum = np.dot(x, weights)
        output = sigmoid(weighted_sum)
        
        # Error
        error = desired - output
        total_error += np.sum(error ** 2)
        
        # Update with Derivative (The "Brake")
        # Even with High Learning Rate, this derivative approaches 0 
        # as the output gets correct, preventing "overshooting".
        sigmoid_derivative = output * (1 - output)
        weights += LEARNING_RATE * np.dot(x.T, error * sigmoid_derivative)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Error: {total_error:.4f}")

# --- REPORT ---
print("\n--- FINAL REPORT ---")
correct_count = 0

print(f"{'Actual':<10} | {'Predicted':<10} | {'Status'}")
print("-" * 35)

for i in range(n_samples):
    x = inputs[i].reshape(1, INPUT_NODES)
    output = sigmoid(np.dot(x, weights))
    
    prediction_index = np.argmax(output)
    predicted_symbol = REVERSE_MAP[prediction_index]
    actual_symbol = REVERSE_MAP[targets[i]]
    
    if predicted_symbol == actual_symbol:
        correct_count += 1
        status = "OK"
    else:
        status = "FAIL"
    
    # Print first 5 and last 5
    if i < 5 or i >= n_samples - 5:
        print(f"{actual_symbol:<10} | {predicted_symbol:<10} | {status}")

print("-" * 35)
print(f"Total Successful Detections: {correct_count} / {n_samples}")
print(f"Accuracy: {(correct_count/n_samples)*100:.2f}%")