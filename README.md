# Custom Neural Network from Scratch (Decimal Precision)

A multi-layer perceptron (MLP) built completely from scratch **without high-level deep learning frameworks** (such as PyTorch or TensorFlow).

The network is trained to perform addition of single-digit numbers ($0+0$ through $9+9$) using pure math and Python's `decimal` module for high-precision arithmetic.

## 🧠 Architecture
* **Input Layer:** 20 neurons (One-Hot encoding for two numbers from 0 to 9: $10 + 10$).
* **Hidden Layer:** 32 neurons with ReLU activation function.
* **Output Layer:** 19 neurons (possible sums ranging from 0 to 18).
* **Key Feature:** Built with Python's `decimal` module using fixed precision to prevent standard `float` rounding artifacts during backpropagation.

## 🛠️ Tech Stack
* Python 3
* NumPy (for initial matrix structure)
* `decimal` (for high-precision gradient and loss calculations)

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ItsRealAwi/Simple-Neural-Network-For-Addting-Two-Numbers.git](https://github.com/ItsRealAwi/Simple-Neural-Network-For-Addting-Two-Numbers.git)
