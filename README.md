# NeuroForge

## Overview

NeuroForge provides custom implementations of MLP models without relying on high-level deep learning frameworks for the core neural network functionality. The project demonstrates fundamental neural network concepts including forward propagation, backpropagation, and various optimization algorithms.

## Project Structure

```
├── MLPClassification.ipynb    # Image classification using MLP
├── MLPRegressor.ipynb        # House price prediction using MLP  
├── multiLabel_Final.ipynb    # Multi-label text classification using MLP
└── README.md                 # Project documentation
```

## Features

### Core MLP Implementation
- **Custom Neural Networks**: Built from scratch using NumPy/PyTorch
- **Flexible Architecture**: Configurable hidden layers and neurons
- **Multiple Activations**: ReLU, Sigmoid, Tanh support
- **Optimization Algorithms**: SGD, Batch GD, Mini-batch GD
- **Weight Initialization**: He, Xavier, and random initialization

### Task-Specific Implementations

#### 1. Image Classification (`MLPClassification.ipynb`)
- **Dataset**: Symbol recognition with 32x32 grayscale images
- **Features**: Image preprocessing, 10-fold cross-validation
- **Architecture**: Input(1024) → Hidden(1024, 512) → Output(num_classes)
- **Performance**: Comprehensive hyperparameter tuning across multiple folds

#### 2. Regression (`MLPRegressor.ipynb`)
- **Dataset**: Bangalore house price prediction
- **Features**: Data cleaning, feature engineering, outlier removal
- **Preprocessing**: One-hot encoding, standardization
- **Metrics**: MSE, RMSE, R² score evaluation

#### 3. Multi-Label Classification (`multiLabel_Final.ipynb`)
- **Dataset**: News article categorization
- **Features**: TF-IDF vectorization, custom text preprocessing
- **Metrics**: Hamming loss, subset accuracy
- **Text Processing**: Stopword removal, vocabulary building

## Key Components

### Neural Network Architecture
```python
class MLP:
    def __init__(self, input_size, hidden_layers, output_size, 
                 activation='relu', learning_rate=0.01, optimizer='sgd'):
        # Configurable MLP with flexible architecture
```

### Supported Features
- **Activation Functions**: ReLU, Sigmoid, Tanh
- **Optimizers**: Stochastic GD, Batch GD, Mini-batch GD
- **Loss Functions**: Cross-entropy, Binary cross-entropy, MSE
- **Metrics**: Accuracy, Hamming loss, R², MSE, RMSE

## Results Summary

### Classification Performance
- **Best Configuration**: ReLU activation with mini-batch gradient descent
- **Cross-Validation**: Consistent performance across 10 folds
- **Hyperparameter Tuning**: Systematic evaluation of activation functions, optimizers, and learning rates

### Regression Performance
- **Architecture**: [1024, 512, 64] hidden layers achieved best results
- **Optimization**: Mini-batch GD with ReLU activation
- **Metrics**: Achieved competitive R² scores on house price prediction

### Multi-Label Classification
- **Text Processing**: Custom TF-IDF implementation with stopword filtering
- **Performance**: Balanced Hamming loss and subset accuracy
- **Architecture**: [256, 128, 64] layers with sigmoid output for multi-label prediction

## Installation & Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python nltk
```

### Running the Models

1. **Classification**:
   ```python
   # Load and preprocess image data
   X_train, y_train = load_data("path/to/train.csv")
   
   # Initialize and train MLP
   mlp = MLP(input_size=1024, hidden_layers=[1024, 512], 
             output_size=num_classes, activation='relu')
   mlp.train(X_train, y_train, epochs=10)
   ```

2. **Regression**:
   ```python
   # Initialize regressor
   mlp_reg = MLPRegressor(input_size=features, hidden_layers=[128, 64],
                          learning_rate=0.01, optimizer='mini-batch')
   mlp_reg.train(X_train, y_train, epochs=100)
   ```

3. **Multi-Label Classification**:
   ```python
   # Preprocess text data
   X_train, X_val, Y_train, Y_val, vocab, label_mapping = preprocess_data(csv_path)
   
   # Train multi-label MLP
   mlp = MLP(input_size=X_train.shape[1], output_size=Y_train.shape[1],
             hidden_layers=[256, 128, 64])
   history = mlp.train(X_train, Y_train, X_val=X_val, Y_val=Y_val)
   ```

## Hyperparameter Tuning

The project includes comprehensive hyperparameter optimization:

- **Grid Search**: Systematic evaluation of parameter combinations
- **Cross-Validation**: Robust performance estimation
- **Visualization**: Training curves and performance heatmaps
- **Statistical Analysis**: Mean and standard deviation across folds

## Key Findings

1. **Activation Functions**: ReLU generally outperformed sigmoid and tanh
2. **Optimization**: Mini-batch gradient descent provided best balance of speed and stability
3. **Architecture**: Deeper networks (3-4 layers) showed better performance for complex tasks
4. **Consistency**: Lower standard deviation in cross-validation indicated better generalization
