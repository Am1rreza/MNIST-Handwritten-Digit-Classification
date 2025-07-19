# 🧠 MNIST Handwritten Digit Classification (NumPy Implementation)

This project implements a simple feedforward artificial neural network (ANN) from scratch using only NumPy — no TensorFlow, PyTorch, or scikit-learn.

It is trained on the famous MNIST dataset of handwritten digits and achieves **~96.87% accuracy** on the test set.  
The network consists of a single hidden layer and uses the sigmoid activation function, along with gradient descent-based weight updates.

This project is perfect for learning and demonstrating how a basic neural network works under the hood.

## 🗃️ Dataset & Visualization

The model is trained on the **MNIST dataset**, a classic benchmark in computer vision, consisting of 70,000 grayscale images of handwritten digits (0 to 9). Each image is 28×28 pixels, flattened into a 784-dimensional vector.

- `mnist_train.csv` – contains 60,000 labeled training examples  
- `mnist_test.csv` – contains 10,000 labeled test examples

### 🔍 Data Sample Visualization

To verify the data and visually understand the digits, a function `array_to_image(index)` is provided to display any specific sample from the training set.

```python
array_to_image(0)
```
This will show a grayscale image along with its actual label.

## 🧠 Neural Network Architecture

This project features a fully connected feedforward neural network with the following architecture:

- **Input Layer**: 784 nodes (28×28 pixels)
- **Hidden Layer**: 100 nodes
- **Output Layer**: 10 nodes (one for each digit class: 0–9)
- **Activation Function**: Sigmoid (`expit` from `scipy.special`)
- **Weight Initialization**: Normal distribution scaled by number of incoming nodes
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Manual weight update using gradient descent (backpropagation)

The class `NeuralNetwork` includes three key methods:

- `__init__()` – initializes weights and sets learning rate  
- `train()` – performs forward pass, calculates error, and updates weights  
- `query()` – performs forward pass and returns predicted outputs
- 
## 🏋️ Training the Network

Before training, the pixel values are scaled to be between **0.01 and 1.0**. This helps avoid issues with zero inputs in the sigmoid activation function.

The network is trained for **10 epochs** on the entire training dataset.

For each training example:

- The input is a normalized 784-dimensional vector
- The target output is a 10-dimensional vector:
  - All values are `0.01`
  - The correct digit's index is set to `0.99`

Example target for digit 5:
```python
[0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01]
```
The training loop updates weights using gradient descent after each example (stochastic training).

## 🧪 Testing & Evaluation

After training, the network is evaluated on the **mnist_test.csv** dataset.

For each test image:

- The network performs a forward pass using `query()`
- The predicted label is selected using `np.argmax(outputs)`
- The prediction is compared to the true label

The final accuracy is calculated as:

```python
accuracy = (correct_predictions / total_predictions) * 100
```
✅ Final Accuracy:
The trained network achieved an accuracy of:

🎯 96.87%

This is a solid result for a network trained from scratch using only NumPy!

## ✅ Conclusion & Future Work

This project demonstrates how a basic neural network can be implemented **from scratch** in Python using only NumPy, and still achieve strong results on a well-known dataset.

### 📌 Key Takeaways:
- Achieved 96.87% accuracy on MNIST without using deep learning libraries.
- Learned the fundamentals of forward propagation, backpropagation, weight updates, and activation functions.

### 🚀 Potential Improvements:
- Add more hidden layers to make it a deeper network.
- Use different activation functions like ReLU or tanh.
- Implement momentum or Adam optimizer.
- Add support for cross-entropy loss instead of MSE.
- Experiment with dropout or batch normalization.

This project provides a solid foundation for those who want to build an intuition for how neural networks actually work behind the scenes.
