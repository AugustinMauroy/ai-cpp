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

    neuralNetwork.train(trainingData, 10000000);
    
    std::vector<std::vector<double>> testData = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    for (auto& input : testData) {
        auto output = neuralNetwork.feedforward(input);
        bool correct = (output[0] < 0.5 && input[0] == 0.0 && input[1] == 0.0) ||
                       (output[0] > 0.5 && input[0] == 0.0 && input[1] == 1.0) ||
                       (output[0] > 0.5 && input[0] == 1.0 && input[1] == 0.0) ||
                       (output[0] < 0.5 && input[0] == 1.0 && input[1] == 1.0);
        std::cout  << "\033[0m Input: \033[1m" << input[0] << " " << input[1] << "\033[0m " << "Output: " << (correct ? "\033[1;32m" : "\033[1;31m") << output[0]  << std::endl;
    }

    return 0;
}
