#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>
#include "src/nn.cpp"

std::vector<std::string> read_label_names(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<std::string> label_names;
    if (file.is_open()) {
        std::string label;
        while (std::getline(file, label)) {
            label_names.push_back(label);
        }
        file.close();
    } else {
        std::cerr << "Error: failed to open file " << file_path << std::endl;
    }
    return label_names;
}

std::vector<std::string> coarse_label_names = read_label_names("dataset/cifar-100-binary/coarse_label_names.txt");
std::vector<std::string> fine_label_names = read_label_names("dataset/cifar-100-binary/fine_label_names.txt");

void display_image(const std::vector<uint8_t>& image) {
    // clear the screen
    std::cout << "\033[2J";
    // move the cursor to the top left corner
    std::cout << "\033[0;0H";
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 32; ++j) {
            uint8_t r = image[i * 32 + j];
            uint8_t g = image[1024 + i * 32 + j];
            uint8_t b = image[2048 + i * 32 + j];
            // convert RGB to ansii color code
            std::string color_code = "\033[48;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m  ";
            std::cout << color_code;
        }
        // reset the color
        std::cout << "\033[0m" << std::endl;
    }
    std::cout << std::endl;
}

// read CIFAR-100 dataset from binary file
std::vector<std::pair<std::vector<uint8_t>, std::pair<uint8_t, uint8_t>>> read_cifar100(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: failed to open file " << file_path << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    ProgressBar progress_bar(file_size);
    std::vector<std::pair<std::vector<uint8_t>, std::pair<uint8_t, uint8_t>>> data;
    while (file.tellg() < file_size) {
        std::vector<uint8_t> image(3072);
        uint8_t coarse_label, fine_label;
        file.read(reinterpret_cast<char*>(&coarse_label), 1);
        file.read(reinterpret_cast<char*>(&fine_label), 1);
        file.read(reinterpret_cast<char*>(image.data()), 3072);
        // Push the image and label pair into the data vector
        data.push_back({image, {coarse_label, fine_label}});
        progress_bar.update();

        // debug
        // display_image(image);
        // std::cout << "Coarse label: " << coarse_label_names[coarse_label] << std::endl;
        // std::cout << "Fine label: " << fine_label_names[fine_label] << std::endl;
        // std::cin.get();
    }

    return data;
}




int main(void) {
    std::cout << "\033[33;1mWARNING:\033[0m This program isn't 100\% accurate, I (Augustin) can't guarantee the accuracy of the results." << std::endl;
    std::string train_file_path = "dataset/cifar-100-binary/train.bin";
    std::string test_file_path = "dataset/cifar-100-binary/test.bin";

    std::cout << "Reading CIFAR-100 train dataset..." << std::endl;
    auto train_data = read_cifar100(train_file_path);
    std::cout << "Reading CIFAR-100 test dataset..." << std::endl;
    auto test_data = read_cifar100(test_file_path);

    NeuralNetworkConfig config;
    config.inputSize = 3072;
    config.hiddenSize = 100;
    config.outputSize = 100;
    config.learningRate = 0.1;
    config.activationFunction = ActivationFunction::RELU;

    NeuralNetwork cifar100_network(config, config.activationFunction);

    int modelLoaded = cifar100_network.loadModel("cifar100-model.txt");

    if (!modelLoaded) {
        std::cout << "Loading CIFAR-100 traning data..." << std::endl;
        std::vector<std::pair<std::vector<double>, std::vector<double>>> cifar100_training_data;
        for (size_t i = 0; i < train_data.size(); ++i) {
            std::vector<double> pixelValues(3072, 0.0);
            for (size_t j = 0; j < 3072; ++j) {
                pixelValues[j] = static_cast<double>(train_data[i].first[j]) / 255.0;
            }
            std::vector<double> target(255, 0.0);
            target[train_data[i].second.first] = 1.0;
            cifar100_training_data.push_back({pixelValues, target});
        }
        std::cout << "Training CIFAR-100 neural network..." << std::endl;
        cifar100_network.train(cifar100_training_data, cifar100_training_data, 10000);
        std::cout << "Saving CIFAR-100 neural network model..." << std::endl;
        cifar100_network.saveModel("cifar100-model.txt");
    }

    std::cout << "Testing CIFAR-100 neural network..." << std::endl;

    int correct_predictions = 0;
    ProgressBar progress_bar(test_data.size());
    for (size_t i = 0; i < test_data.size(); ++i) {
        progress_bar.update();
        std::vector<double> pixelValues(3072, 0.0);
        for (size_t j = 0; j < 3072; ++j) {
            pixelValues[j] = static_cast<double>(test_data[i].first[j]) / 255.0;
        }
        std::vector<double> prediction = cifar100_network.feedforward(pixelValues);
        size_t max_index = 0;
        for (size_t j = 1; j < prediction.size(); ++j) {
            if (prediction[j] > prediction[max_index]) {
                max_index = j;
            }
        }
        if (max_index == test_data[i].second.first) {
            correct_predictions++;
        }
    }

    std::cout << "Accuracy: " << static_cast<double>(correct_predictions) / test_data.size() << std::endl;

    while (true) {
        std::cout << "Enter an image index to test (0-" << test_data.size() - 1 << "): ";
        size_t index;
        std::cin >> index;
        if (index >= test_data.size()) {
            std::cout << "Invalid index" << std::endl;
            continue;
        }
        display_image(test_data[index].first);
        std::cout << "Label: " << fine_label_names[test_data[index].second.first] << std::endl;
        std::vector<double> pixelValues(3072, 0.0);
        for (size_t j = 0; j < 3072; ++j) {
            pixelValues[j] = static_cast<double>(test_data[index].first[j]) / 255.0;
        }
        std::vector<double> prediction = cifar100_network.feedforward(pixelValues);
        size_t max_index = 0;
        for (size_t j = 1; j < prediction.size(); ++j) {
            if (prediction[j] > prediction[max_index]) {
                max_index = j;
            }
        }
        std::cout << "Prediction: " << fine_label_names[max_index] << std::endl;
        std::cout << "Press enter to continue...";
        std::cin.get();
        std::cin.get();
    }
    return EXIT_SUCCESS;
}