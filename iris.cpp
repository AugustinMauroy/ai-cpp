#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include "src/nn.cpp"

std::vector<std::pair<std::vector<double>, std::vector<double>>> loadIrisData(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::pair<std::vector<double>, std::vector<double>>> data;

    if (file.is_open()) {
        std::string line;
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string token;
            std::vector<double> features;
            std::vector<double> label(3, 0.0);

            for (int i = 0; i < 4; ++i) {
                std::getline(iss, token, ',');
                features.push_back(std::stod(token));
            }

            std::getline(iss, token, ',');
            std::getline(iss, token, ',');
            if (token == "\"Setosa\"")
                label[0] = 1.0;
            else if (token == "\"Versicolor\"")
                label[1] = 1.0;
            else if (token == "\"Virginica\"")
                label[2] = 1.0;

            data.push_back({features, label});
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return data;
}

int main(void) {
    NeuralNetworkConfig config;
    config.inputSize = 4;
    config.hiddenSize = 8;
    config.outputSize = 3;
    config.learningRate = 0.1; 
    config.activationFunction = ActivationFunction::SIGMOID;

    NeuralNetwork neuralNetwork(config, ActivationFunction::SIGMOID);

    int modelLoaded = neuralNetwork.loadModel("iris-model.txt");

    std::vector<std::pair<std::vector<double>, std::vector<double>>> irisData = loadIrisData("./dataset/iris.csv");

    if (!modelLoaded) {
        std::cout << "Model not found. Training neural network..." << std::endl;
        neuralNetwork.train(irisData, irisData, 1000000);
        std::cout << "Training complete." << std::endl;

        neuralNetwork.saveModel("iris-model.txt");
    }

    int correctPredictions = 0;
    for (const auto& [features, label] : irisData) {
        std::vector<double> outputs = neuralNetwork.feedforward(features);
        int predictedClass = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
        int trueClass = std::distance(label.begin(), std::max_element(label.begin(), label.end()));
        if (predictedClass == trueClass) {
            correctPredictions++;
        }
    }

    std::cout << "Accuracy: " << static_cast<double>(correctPredictions) / irisData.size() * 100 << "%" << std::endl;

    std::vector<double> features = { 6.1, 2.9, 4.7, 1.4 };
    std::vector<double> outputs = neuralNetwork.feedforward(features);
    int predictedClass = std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));

    std::cout << "Features: " << std::endl;
    std::cout << "Sepal length: \033[1;32m"<< features[0] << "\033[0m" << std::endl;
    std::cout << "Sepal width: \033[1;32m" << features[1] << "\033[0m" << std::endl;
    std::cout << "Petal length: \033[1;32m" << features[2] << "\033[0m" << std::endl;
    std::cout << "Petal width: \033[1;32m" << features[3] << "\033[0m" << std::endl;
    std::cout << "Predicted class: ";
    if (predictedClass == 0)
        std::cout << "Setosa" << std::endl;
    else if (predictedClass == 1)
        std::cout << "Versicolor" << std::endl;
    else if (predictedClass == 2)
        std::cout << "Virginica" << std::endl;

    return EXIT_SUCCESS;
}