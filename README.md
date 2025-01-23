# Hopfield Network

Implementacja i analiza sieci Hopfielda na przykładzie rozpoznawania wzorców odręcznie zapisanych cyfr (zbiór danych MNIST).

## Overview

Sieć Hopfielda tworzona jest za według reguły Hebba, dostępne są 2 tryby predykcji: synchroniczny i asynchroniczny. Sieć zapamiętuje 5 wzorców.

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
```
