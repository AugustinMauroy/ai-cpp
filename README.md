# AI-CPP

A simple neural network implementation in C++.

> \[!IMPORTANT]\
> This project is learning/discovering oriented.

## Build

This project require [LLVM](https://llvm.org/) installed on your system.

> \[!NOTE]\
> It's build with standard C++17 library. So you don't need to install any other library.

```bash
# xor example
g++ -std=c++17 -o xor_neural_network xor.cpp && ./xor_neural_network
# angles example
g++ -std=c++17 -o angles_neural_network angles.cpp && ./angles_neural_network
# mnist example
g++ -std=c++17 -o mnist_neural_network mnist.cpp && ./mnist_neural_network
# iris example
g++ -std=c++17 -o iris_neural_network iris.cpp && ./iris_neural_network
# cifar-100 example need curl and tar
cd dataset
curl -O https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar -xvf cifar-100-binary.tar.gz
rm cifar-100-binary.tar.gz
cd ../
g++ -std=c++17 -o cifar-100_neural_network cifar-100.cpp && ./cifar-100_neural_network
```

## Neural Network lib

- [doc](/docs/nn.md)
- [code](/src/nn.cpp)
