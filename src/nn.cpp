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
};

struct NeuralNetworkConfig {
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
};

class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    double learningRate;
    struct {
        std::vector<std::vector<double> > inputToHidden;
        std::vector<std::vector<double> > hiddenToOutput;
    } weights;

public:
    NeuralNetwork(const NeuralNetworkConfig& config) {
        inputSize = config.inputSize;
        hiddenSize = config.hiddenSize;
        outputSize = config.outputSize;
        learningRate = config.learningRate;

        weights.inputToHidden.resize(inputSize, std::vector<double>(hiddenSize));
        weights.hiddenToOutput.resize(hiddenSize, std::vector<double>(outputSize));

        // Initialize the weights of the neural network with random values
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(0.0, 1.0);

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

    std::vector<double> feedforward(const std::vector<double>& inputs) {
        std::vector<double> hiddenOutputs(hiddenSize, 0.0);

        // Calculate the outputs of the hidden layer
        for (int i = 0; i < hiddenSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < inputSize; j++) {
                sum += inputs[j] * weights.inputToHidden[j][i];
            }
            hiddenOutputs[i] = MathUtils::sigmoid(sum);
        }

        std::vector<double> outputs(outputSize, 0.0);

        // Calculate the outputs of the output layer
        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenOutputs[j] * weights.hiddenToOutput[j][i];
            }
            outputs[i] = MathUtils::sigmoid(sum);
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
            hiddenOutputs[i] = MathUtils::sigmoid(sum);
        }

        for (int i = 0; i < outputSize; i++) {
            double sum = 0.0;
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenOutputs[j] * weights.hiddenToOutput[j][i];
            }
            outputs[i] = MathUtils::sigmoid(sum);
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

    void test(const std::vector<double>& input) {
        // Test the neural network
        std::vector<double> outputs = feedforward(input);
        std::cout << "Input: \033[1;32m" << input[0] << " " << input[1] << "\033[0m | Output: \033[1;32m" << outputs[0] << "\033[0m" << std::endl;
    }

    void saveModel(const std::string& filePath) {
        std::ofstream file(filePath);
        if (file.is_open()) {
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    file << weights.inputToHidden[i][j] << " ";
                }
            }
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    file << weights.hiddenToOutput[i][j] << " ";
                }
            }
            file.close();
        }
    }

    void loadModel(const std::string& filePath) {
        std::ifstream file(filePath);
        if (file.is_open()) {
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    file >> weights.inputToHidden[i][j];
                }
            }
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    file >> weights.hiddenToOutput[i][j];
                }
            }
            file.close();
        }
    }
};