# Deep Learning from Scratch (PyTorch)

This folder contains basic implementations of various machine learning and deep learning models, built from scratch using PyTorch. The goal of these implementations is to understand the fundamentals behind these models.

## Contents

### Python Scripts

- **`neuron.py`**
  - Implements a single artificial neuron.
  - Includes manual forward pass and training loop using gradient descent.

- **`mlp.py`**
  - Implements a Multi-Layer Perceptron (MLP).
  - Defines a `Layer` class and an `MLP` class that stacks multiple layers.
  - Supports forward pass and training with backpropagation.

- **`nanoGPT.py`**
  - A minimal implementation of a GPT-style language model (NanoGPT).
  - Trained on the Tiny Shakespeare dataset.
  - Includes implementations of:
    - `SelfAttention`
    - `MultiHeadedAttention`
    - `FeedForwardBlock`
    - `GPT` model architecture
  - Handles data loading, tokenization, and batch generation.

### Jupyter Notebooks

- **`cnn_mnist.ipynb`**
  - Implements a Convolutional Neural Network (CNN) for the MNIST dataset.
  - Features a custom `Convolver` class that performs convolution operations using matrix multiplication (unfolding/im2col technique) rather than using PyTorch's built-in `nn.Conv2d` directly for the core operation.

- **`Iris_flower_classifier.ipynb`**
  - Applies the Multi-Layer Perceptron (MLP) architecture to the Iris flower classification problem.
  - Demonstrates data loading with Pandas and training a custom MLP model.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib

