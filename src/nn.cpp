#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>

class MathUtils {
public:
    static double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
    static double tanh(double x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }
};

enum ActivationFunction {
    TANH,
    SIGMOID
};

struct NeuralNetworkConfig {
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
    ActivationFunction activationFunction;
};

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
    ActivationFunction activationFunction;

    struct {
        std::vector<std::vector<double>> inputToHidden;
        std::vector<std::vector<double>> hiddenToOutput;
    } weights;

public:
    NeuralNetwork(const NeuralNetworkConfig& config, ActivationFunction activationFunction)
        : inputSize(config.inputSize), hiddenSize(config.hiddenSize),
          outputSize(config.outputSize), learningRate(config.learningRate),
          activationFunction(activationFunction) {

        // Initialize the weights of the neural network with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        weights.inputToHidden = std::vector<std::vector<double>>(inputSize, std::vector<double>(hiddenSize));
        weights.hiddenToOutput = std::vector<std::vector<double>>(hiddenSize, std::vector<double>(outputSize));

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights.inputToHidden[i][j] = dist(gen);
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights.hiddenToOutput[i][j] = dist(gen);
            }
        }
    }

    double activate(double x) {
        switch (activationFunction) {
        case SIGMOID:
            return MathUtils::sigmoid(x);
            break;
        case TANH:
            return MathUtils::tanh(x);
            break;
        default:
            MathUtils::tanh(x);
            break;
        }
    }

    std::vector<double> feedforward(const std::vector<double>& inputs) {
        std::vector<double> hiddenOutputs(hiddenSize, 0.0);

        // Calculate the outputs of the hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weights.inputToHidden[j][i];
            }
            hiddenOutputs[i] = activate(sum);
        }

        std::vector<double> outputs(outputSize, 0.0);

        // Calculate the outputs of the output layer
        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenOutputs[j] * weights.hiddenToOutput[j][i];
            }
            outputs[i] = activate(sum);
        }

        return outputs;
    }

    void backpropagation(const std::vector<double>& inputs, const std::vector<double>& targets) {
        std::vector<double> hiddenOutputs(hiddenSize, 0.0);
        std::vector<double> outputs(outputSize, 0.0);

        // Calculate the outputs of the hidden layer and the final output
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weights.inputToHidden[j][i];
            }
            hiddenOutputs[i] = activate(sum);
        }

        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenOutputs[j] * weights.hiddenToOutput[j][i];
            }
            outputs[i] = activate(sum);
        }

        // Calculate the output error
        std::vector<double> outputErrors(outputSize, 0.0);
        for (int i = 0; i < outputSize; i++) {
            outputErrors[i] = targets[i] - outputs[i];
        }

        // Calculate the hidden layer error
        std::vector<double> hiddenErrors(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < outputSize; j++) {
                sum += outputErrors[j] * weights.hiddenToOutput[i][j];
            }
            hiddenErrors[i] = hiddenOutputs[i] * (1.0 - hiddenOutputs[i]) * sum;
        }

        // Update the weights from the hidden layer to the output
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights.hiddenToOutput[i][j] += learningRate * outputErrors[j] * hiddenOutputs[i];
            }
        }

        // Update the weights from the input to the hidden layer
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weights.inputToHidden[i][j] += learningRate * hiddenErrors[j] * inputs[i];
            }
        }
    }

    void train(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& trainingData, int numberOfIterations) {
        // Train the neural network
        for (int i = 0; i < numberOfIterations; i++) {
            int randomIndex = rand() % trainingData.size();
            const auto& [randomInputs, randomTargets] = trainingData[randomIndex];
            backpropagation(randomInputs, randomTargets);
        }
    }

    void saveModel(const std::string& filePath) {
        std::ofstream file(filePath);
        if (file.is_open()) {
            for (const auto& row : weights.inputToHidden) {
                for (const double val : row) {
                    file << val << ' ';
                }
            }
            for (const auto& row : weights.hiddenToOutput) {
                for (const double val : row) {
                    file << val << ' ';
                }
            }
            file.close();
        } else {
            std::cout << "Unable to open file " << filePath << std::endl;
        }
    }

    int loadModel(const std::string& filePath) {
        std::ifstream file(filePath);
        if (file.is_open()) {
            for (auto& row : weights.inputToHidden) {
                for (double& val : row) {
                    file >> val;
                }
            }
            for (auto& row : weights.hiddenToOutput) {
                for (double& val : row) {
                    file >> val;
                }
            }
            file.close();
            return true;
        } else {
            std::cout << "No model found at " << filePath << std::endl;
            return false;
        }
    }
};

#endif
