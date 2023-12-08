#include <iostream>
#include <iomanip>

class ProgressBar {
public:
    ProgressBar(int total, int width = 50) : total(total), width(width), progress(0) {}

    void update() {
        ++progress;

        float percentage = static_cast<float>(progress) / total;
        int currentWidth = static_cast<int>(percentage * width);

        std::cout << "[";
        for (int i = 0; i < currentWidth; ++i) {
            std::cout << "=";
        }
        for (int i = currentWidth; i < width; ++i) {
            std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (percentage * 100.0) << "%\r";
        std::cout.flush();

        if (progress == total) {
            std::cout << std::endl;  // Move to the next line after completion
        }
    }

private:
    int total;
    int width;
    int progress;
};
