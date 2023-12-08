# Neural Network C++ Implementation Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [MathUtils Class](#mathutils-class)
3. [Activation Functions](#activation-functions)
4. [Neural Network Configuration](#neural-network-configuration)
5. [Neural Network Class](#neural-network-class)
    - [Constructor](#constructor)
    - [Activation Function](#activation-function)
    - [Feedforward](#feedforward)
    - [Backpropagation](#backpropagation)
    - [Training](#training)
    - [Model Saving and Loading](#model-saving-and-loading)

## Introduction

The `NeuralNetwork` C++ implementation provides a flexible and customizable framework for creating and training feedforward neural networks. The implementation supports various activation functions, including sigmoid, hyperbolic tangent (tanh), rectified linear unit (ReLU), linear, and softmax. The neural network is designed to handle a configurable number of input, hidden, and output nodes.

## MathUtils Class

The `MathUtils` class contains static methods for common mathematical operations used in neural networks. Currently, it provides methods for calculating the sigmoid and hyperbolic tangent functions.

### Sigmoid Function

```cpp
static double sigmoid(double x);
```

### Hyperbolic Tangent Function

```cpp
static double tanh(double x);
```

## Activation Functions

The `ActivationFunction` enumeration defines the supported activation functions for the neural network. The available functions include:

- `TANH`
- `SIGMOID`
- `RELU`
- `LINEAR`
- `TANH_DERIVATIVE`
- `SOFTMAX`

## Neural Network Configuration

The `NeuralNetworkConfig` struct encapsulates the configuration parameters for creating a neural network. These parameters include:

- `inputSize`: Number of input nodes
- `hiddenSize`: Number of hidden nodes
- `outputSize`: Number of output nodes
- `learningRate`: Learning rate for weight updates during training
- `activationFunction`: Activation function for the hidden and output layers

## Neural Network Class

The `NeuralNetwork` class encapsulates the functionality of a feedforward neural network.

### Constructor

```cpp
NeuralNetwork(const NeuralNetworkConfig& config, ActivationFunction activationFunction);
```

- **Parameters:**
  - `config`: Configuration parameters for the neural network.
  - `activationFunction`: Activation function for hidden and output layers.

### Activation Function

```cpp
double activate(double x);
```
- **Parameters:**
  - `x`: Input value to the activation function.
- **Returns:**
  - The result of applying the specified activation function to the input.

### Feedforward

```cpp
std::vector<double> feedforward(const std::vector<double>& inputs);
```
- **Parameters:**
  - `inputs`: Input values to the neural network.
- **Returns:**
  - The output values of the neural network after a feedforward pass.

### Backpropagation

```cpp
void backpropagation(const std::vector<double>& inputs, const std::vector<double>& targets);
```
- **Parameters:**
  - `inputs`: Input values to the neural network.
  - `targets`: Target output values for the given inputs.
- **Description:**
  - Performs backpropagation to update the weights of the neural network.

### Training

```cpp
void train(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainingData, int numberOfIterations);
```

- **Parameters:**
  - `trainingData`: Training data in the form of input-output pairs.
  - `numberOfIterations`: Number of training iterations.
- **Description:**
  - Trains the neural network using the provided training data.

### Model Saving and Loading

```cpp
void saveModel(const std::string& filePath);
```

- **Parameters:**
  - `filePath`: Path to the file where the model will be saved.
- **Description:**
  - Saves the neural network model to a file.

```cpp
int loadModel(const std::string& filePath);
```

- **Parameters:**
  - `filePath`: Path to the file from which the model will be loaded.
- **Returns:**
  - Returns `true` if the model is successfully loaded, otherwise `false`.
- **Description:**
  - Loads a previously saved neural network model from a file.

This C++ implementation provides a foundation for building and experimenting with neural networks, allowing users to customize the architecture, activation functions, and training process based on their specific needs.
