# Native C++ Deep Learning Framework

A custom, dependency-free Deep Learning framework written entirely in standard C++. 
Engineered from the ground up to support both non-linear classification and multi-dimensional regression algorithms dynamically without relying on external libraries like Python's PyTorch or TensorFlow.

## 🚀 Features

- **Object-Oriented Architecture**: Modular `Model`, `Layer`, `Optimizer`, and `Loss` interfaces.
- **Native Matrix Core**: Dynamic memory allocation and 2D tensor abstractions using generic pointer templates (`vector<double>`).
- **OpenMP Multithreading**: C-block accelerated matrix arithmetic via parallelized CPU hardware multithreading (`#pragma omp parallel for`).
- **Memory Optimized**: O(n³) loop calculations are cache-line reordered natively allowing instant vector multiplications (AVX-2).
- **Extensible Neural Layers**: Currently implementing `Dense` structures alongside customizable dynamic activations like `ReLU` and `SoftMax`.
- **Advanced Optimizers**: Custom internal implementation of Adaptive Moment Estimation (`Adam`).
- **Model I/O Integration**: Seamless structural write sequences parsing out dense matrices to binary disk locations for instant inference reloading.

## ⚙️ Build Instructions

This engine runs at maximum speed utilizing strict performance execution flags. Ensure you have `g++` configured on your system.

```bash
g++ -O3 -march=native -mavx2 -fopenmp main.cpp math/*.cpp nn/*.cpp optim/*.cpp data/*.cpp trainer/*.cpp loss/*.cpp metrics/*.cpp scheduler/*.cpp model/*.cpp -o ml.exe
```

## 🧠 Module Architecture

### The Trainer Subsystem
Designed to intake arrays and auto-minimize objectives securely via polymorphic references (`Loss*`). 

```cpp
MSE mse_loss;
Adam adam_optim(0.001);

// Natively tracks gradients on raw custom architectures smoothly!
Trainer trainer(net, adam_optim, &mse_loss);
trainer.fit(train_loader, val_loader, 10);
```

### The Dense Network 
A fully functional `Layer` inheritance abstraction. Propagates gradients backwards flawlessly natively over batches spanning over `200,000+` rows!

```cpp
Model net;
net.add(make_unique<Dense>(10, 64));
net.add(make_unique<ReLU>());
net.add(make_unique<Dense>(64, 32));
// ...
```
