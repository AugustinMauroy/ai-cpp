#include <iostream>
#include <fstream>
#include <vector>
#include "src/nn.cpp"

std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return {};
    }
    
    uint32_t magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    
    // __builtin_bswap32 is a GCC builtin that swaps the bytes of a 32-bit integer
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
    
    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file!" << std::endl;
        return {};
    }
    
    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(num_rows * num_cols));
    for (auto& image : images) {
        file.read(reinterpret_cast<char*>(image.data()), num_rows * num_cols);
    }
    
    return images;
}

std::vector<uint8_t> read_mnist_labels(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return {};
    }
    
    uint32_t magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    
    // __builtin_bswap32 is a GCC builtin that swaps the bytes of a 32-bit integer
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);
    
    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file!" << std::endl;
        return {};
    }
    
    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);
    
    return labels;
}

void display_mnist_image(const std::vector<uint8_t>& image, uint32_t num_rows, uint32_t num_cols) {
    for (uint32_t i = 0; i < num_rows * num_cols; ++i) {
        if (i % num_cols == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << (image[i] > 128 ? "#" : " ");
    }
    std::cout << std::endl;
}

int main(void) {
    std::cout << "MNIST Neural Network" << std::endl;
    std::cout << "\033[33;1mWARNING:\033[0m This program isn't 100\% accurate, it's just a simple neural network implementation." << std::endl;
    std::string images_file_path = "dataset/images/train-images.idx3-ubyte";
    std::string labels_file_path = "dataset/images/train-labels.idx1-ubyte";
    
    std::vector<std::vector<uint8_t>> images = read_mnist_images(images_file_path);
    std::vector<uint8_t> labels = read_mnist_labels(labels_file_path);
    if (images.size() != labels.size()) {
        std::cerr << "Number of images and labels do not match!" << std::endl;
        return 1;
    }

    if (images.empty() || labels.empty()) {
        return 1;
    }
    
    uint32_t num_rows = 28; 
    uint32_t num_cols = 28;
    uint32_t num_images = images.size();
    
    NeuralNetworkConfig config;
    config.inputSize = num_rows * num_cols;
    config.hiddenSize = 128;
    config.outputSize = 10;
    config.learningRate = 0.01;
    config.activationFunction = TANH;

    double dropoutRate = 0.2;
    NeuralNetwork mnistNetwork(config, config.activationFunction, dropoutRate);

    int modelLoaded = mnistNetwork.loadModel("mnist-model.txt");

    if (!modelLoaded) {
        std::cout << "Loading MNIST traning data..." << std::endl;
        std::vector<std::pair<std::vector<double>, std::vector<double>>> mnistTrainingData;
        for (size_t i = 0; i < num_images; ++i) {
            std::vector<double> pixelValues(num_rows * num_cols, 0.0);
            for (size_t j = 0; j < num_rows * num_cols; ++j) {
                pixelValues[j] = static_cast<double>(images[i][j]) / 255.0;
            }
            std::vector<double> target(10, 0.0);
            target[labels[i]] = 1.0;
            mnistTrainingData.push_back({pixelValues, target});
        }
        std::cout << "MNIST data loaded !" << std::endl;

        std::cout << "Training neural network..." << std::endl;
        mnistNetwork.train(mnistTrainingData, mnistTrainingData, 100);
        std::cout << "Training complete." << std::endl;

        mnistNetwork.saveModel("mnist-model.txt");
    }

    int correctPredictions = 0;
    std::cout << "Testing neural network..." << std::endl;
    ProgressBar progressBar(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        progressBar.update();
        std::vector<double> pixelValues(num_rows * num_cols, 0.0);
        for (size_t j = 0; j < num_rows * num_cols; ++j) {
            pixelValues[j] = static_cast<double>(images[i][j]) / 255.0;
        }
        std::vector<double> output = mnistNetwork.feedforward(pixelValues);
        int predictedLabel = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predictedLabel == static_cast<int>(labels[i])) {
            correctPredictions++;
        }
    }

    double accuracy = static_cast<double>(correctPredictions) / num_images * 100.0;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // part to allow user to test the model
    while (true) {
        std::cout << "Enter the index of the image you want to test (0-" << num_images - 1 << "): ";
        size_t index;
        std::cin >> index;
        if (index >= num_images) {
            std::cerr << "Invalid index!" << std::endl;
            continue;
        }
        std::vector<double> pixelValues(num_rows * num_cols, 0.0);
        for (size_t j = 0; j < num_rows * num_cols; ++j) {
            pixelValues[j] = static_cast<double>(images[index][j]) / 255.0;
        }
        std::vector<double> output = mnistNetwork.feedforward(pixelValues);
        std::cout << "Predicted label: " << std::distance(output.begin(), std::max_element(output.begin(), output.end())) << std::endl;
        std::cout << "Actual label: " << static_cast<int>(labels[index]) << std::endl;
        display_mnist_image(images[index], num_rows, num_cols);
    }

    return EXIT_SUCCESS;
}
