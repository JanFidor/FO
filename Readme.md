# Hopfield Network MNIST Classifier

A Python implementation of a Hopfield Network for pattern recognition using the MNIST dataset.

## Overview
This project implements a Hopfield Network to recognize and reconstruct noisy MNIST handwritten digits. The network can learn patterns from the MNIST dataset and approximate their recovery when given noisy versions.

## Features
- MNIST dataset loading and preprocessing
- Configurable noise levels
- Both synchronous and asynchronous network updates
- Visualizations:
  - Network predictions
  - Energy transitions
  - Weight matrix evolution (gif)

## Usage
```python
import mnist
import main

# Load MNIST data with different noise levels
fetch = mnist.fetch_minist_for_hopfield(size=5, error_rates=[0.1, 0.14, 0.25, 0.3, 0.5])

# Run prediction with specified bias
main.save_mnist_prediction(fetch=fetch[0], bias=60, sync=False)