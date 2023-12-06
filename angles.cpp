#include <iostream>
#include "./src/nn.cpp"

int main() {
    int inputSize = 2;
    int hiddenSize = 8;
    int outputSize = 4;
    double learningRate = 0.01;
    ActivationFunction activation = TANH;

    NeuralNetworkConfig config = {inputSize, hiddenSize, outputSize, learningRate};
    NeuralNetwork neuralNetwork(config, activation);
    int modelLoaded = neuralNetwork.loadModel("angles-model.txt");


    if(!modelLoaded) {
        std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData;
        std::cout << "Generating training data..." << std::endl;

        for (double angle = 0; angle < 360; angle += 5) {
            double radians = angle * M_PI / 180.0;
            int quadrant = static_cast<int>(angle / 90.0) % 4 + 1;
            std::vector<double> inputs = {cos(radians), sin(radians)};
            std::vector<double> targets(4, 0.0);
            targets[quadrant - 1] = 1.0;
            trainingData.emplace_back(inputs, targets);
        }
        std::cout << "Training data generated!" << std::endl;

        int numberOfIterations = 10000;
        std::cout << "Starting training..." << std::endl;
        neuralNetwork.train(trainingData, numberOfIterations);
        std::cout << "Training finished!" << std::endl;

        std::cout << "Saving model..." << std::endl;
        neuralNetwork.saveModel("angles-model.txt");
        std::cout << "Model saved!" << std::endl;
    }

    while(1) {
        std::cout << "Enter an angle in degrees (or press ESC to exit): \033[1;32m" << std::flush;
        int key = getchar();
        std::cout << "\033[0m" << std::flush;
        if (key == 27) {
            break;
        }
        std::cin.putback(key);
        double angleToPredict;
        std::cin >> angleToPredict;
        double radiansToPredict = angleToPredict * M_PI / 180.0;
        std::vector<double> inputToPredict = {cos(radiansToPredict), sin(radiansToPredict)};
        std::vector<double> predictedOutputs = neuralNetwork.feedforward(inputToPredict);

        int predictedQuadrant = std::distance(predictedOutputs.begin(), std::max_element(predictedOutputs.begin(), predictedOutputs.end())) + 1;
        std::cout << "L'angle \033[1;32m" << angleToPredict << "\033[0m degrés est prédit être dans le cadran \033[1;32m" << predictedQuadrant << "\033[0m" << std::endl;
    }
    
    return EXIT_SUCCESS;
}
