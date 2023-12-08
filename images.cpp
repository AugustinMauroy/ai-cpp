// TODO: test this neural network on the MNIST dataset
#include <vector>
#include <iostream>
#include <fstream>
#include "src/nn.cpp"
#include "src/progressBar.cpp"

// Function to load MNIST data (you need to have the MNIST dataset files)
std::vector<std::pair<std::vector<double>, std::vector<double>>> loadMNISTData(const std::string& imagesPath, const std::string& labelsPath) {
    std::vector<std::pair<std::vector<double>, std::vector<double>>> mnistData;

    // Read MNIST images
    std::ifstream imagesFile(imagesPath, std::ios::binary);
    std::ifstream labelsFile(labelsPath, std::ios::binary);

    if (imagesFile.is_open() && labelsFile.is_open()) {
        // Read header information
        int magicNumber, numImages, numRows, numCols;
        imagesFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        labelsFile.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        imagesFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        labelsFile.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        imagesFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
        imagesFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        // Normalize pixel values to the range [0, 1]
        const double normalizationFactor = 1.0 / 255.0;
        ProgressBar progressBar(numImages);

        for (int i = 0; i < numImages; ++i) {
            progressBar.update();
            std::vector<double> pixelValues(numRows * numCols, 0.0);

            for (int j = 0; j < numRows * numCols; ++j) {
                unsigned char pixel;
                imagesFile.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                pixelValues[j] = static_cast<double>(pixel) * normalizationFactor;
            }

            unsigned char label;
            labelsFile.read(reinterpret_cast<char*>(&label), sizeof(label));

            // Convert label to one-hot encoding
            std::vector<double> target(10, 0.0);
            target[label] = 1.0;

            mnistData.push_back({pixelValues, target});
        }

        imagesFile.close();
        labelsFile.close();
    } else {
        std::cerr << "Unable to open MNIST files." << std::endl;
    }

    return mnistData;
}

int main(void) {
    // Define the configuration for the neural network
    NeuralNetworkConfig config;
    config.inputSize = 28 * 28;  // 28x28 images
    config.hiddenSize = 64;      // Number of hidden neurons
    config.outputSize = 10;      // 10 classes (digits 0 through 9)
    config.learningRate = 0.01;
    config.activationFunction = ActivationFunction::SIGMOID;

    // Create a neural network with the specified configuration and activation function
    NeuralNetwork mnistNetwork(config, config.activationFunction);

    int modelLoaded = mnistNetwork.loadModel("mnist-model.txt");

    if (!modelLoaded) {
        std::cout << "Loading MNIST traning data..." << std::endl;
        std::vector<std::pair<std::vector<double>, std::vector<double>>> mnistTrainingData =
            loadMNISTData("./dataset/images/train-images.idx3-ubyte", "./dataset/images/train-labels.idx1-ubyte");
        std::cout << "MNIST data loaded !" << std::endl;

        std::cout << "Training neural network..." << std::endl;
        mnistNetwork.train(mnistTrainingData, 10);
        std::cout << "Training complete !" << std::endl;
        mnistNetwork.saveModel("mnist-model.txt");
    }

    std::cout << "Loading MNIST testing data..." << std::endl;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> mnistTestingData =
        loadMNISTData("./dataset/images/t10k-images.idx3-ubyte", "./dataset/images/t10k-labels.idx1-ubyte");
    std::cout << "MNIST testing data loaded !" << std::endl;

    // Test the trained neural network on the MNIST testing data
    int correctPredictions = 0;

    for (const auto& [input, target] : mnistTestingData) {
        std::vector<double> output = mnistNetwork.feedforward(input);

        // Find the index of the maximum value in the output (predicted class)
        int predictedClass = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

        // Find the index of the maximum value in the target (actual class)
        int trueClass = std::distance(target.begin(), std::max_element(target.begin(), target.end()));

        if (predictedClass == trueClass) {
            correctPredictions++;
        }
    }

    // Calculate accuracy
    double accuracy = static_cast<double>(correctPredictions) / mnistTestingData.size();
    std::cout << "Accuracy on MNIST testing data: " << accuracy * 100 << "%" << std::endl;

    return EXIT_SUCCESS;
}
