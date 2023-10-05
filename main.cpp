#include "src/nn.cpp"

int main() {
    NeuralNetworkConfig config;
    config.inputSize = 2;
    config.hiddenSize = 3;
    config.outputSize = 1;
    config.learningRate = 0.1;

    NeuralNetwork neuralNetwork(config);

    // xor example
    std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData = {
        {{0.0, 0.0}, {0.0}},
        {{0.0, 1.0}, {1.0}},
        {{1.0, 0.0}, {1.0}},
        {{1.0, 1.0}, {0.0}}
    };

    neuralNetwork.train(trainingData, 100000);
    neuralNetwork.test({0.0, 0.0});
    neuralNetwork.test({0.0, 1.0});
    neuralNetwork.test({1.0, 0.0});
    neuralNetwork.test({1.0, 1.0});

    return 0;
}
