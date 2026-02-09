import numpy as np
import matplotlib.pyplot as plt

from dataset import X_train, y_train, X_test, y_test, classes, IMAGE_SIZE

class TwoLayerANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate

        scale_w1 = 1 / np.sqrt(input_size)
        scale_w2 = 1 / np.sqrt(hidden_size)
        
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * scale_w1
        self.b1 = np.zeros((1, self.hidden_size))
        
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * scale_w2
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, X, y):
        N = X.shape[0]

        hidden_input = np.dot(X, self.W1) + self.b1
        hidden_output = self.sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.W2) + self.b2
        final_output = self.sigmoid(final_input)

        output_error = y - final_output 
   
        output_delta = output_error * self.sigmoid_derivative(final_output)

        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

        self.W2 += (hidden_output.T.dot(output_delta) / N) * self.lr
        self.b2 += (np.sum(output_delta, axis=0, keepdims=True) / N) * self.lr
        
        self.W1 += (X.T.dot(hidden_delta) / N) * self.lr
        self.b1 += (np.sum(hidden_delta, axis=0, keepdims=True) / N) * self.lr
        
        return np.mean(np.abs(output_error))

    def predict(self, X):
        h_out = self.sigmoid(np.dot(X, self.W1) + self.b1)
        return self.sigmoid(np.dot(h_out, self.W2) + self.b2)


INPUT_NEURONS = IMAGE_SIZE * IMAGE_SIZE 
HIDDEN_NEURONS = 10 
OUTPUT_NEURONS = 4 

ann = TwoLayerANN(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS, learning_rate=2.0)

print(f"Training Model...")

epochs = 10000
error_history = []

for epoch in range(epochs):
    err = ann.train(X_train, y_train)
    error_history.append(err)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Mean Error = {err:.5f}")

print("\n--- FINAL REPORT ---")
correct = 0
total = len(X_test)

for i in range(total):
    prediction = ann.predict(X_test[i:i+1])
    pred_idx = np.argmax(prediction)
    true_idx = np.argmax(y_test[i])
    
    if pred_idx == true_idx:
        correct += 1

print(f"Number of Test Images: {total}")
print(f"Successful Detections: {correct}")
print(f"Final Accuracy: {(correct/total)*100:.2f}%")

plt.figure()
plt.plot(error_history)
plt.title("Error Curve")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()

for i in range(8):
    prediction = ann.predict(X_test[i:i+1])
    pred_idx = np.argmax(prediction)
    true_idx = np.argmax(y_test[i])
    
    img = X_test[i].reshape(IMAGE_SIZE, IMAGE_SIZE)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Pred: {classes[pred_idx]}\nReal: {classes[true_idx]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()