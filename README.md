# Universal Native C++ Deep Learning Framework

A custom, dependency-free Deep Learning framework written entirely in standard C++. 
Engineered from the ground up to support uploading any raw `.csv` datasheet dynamically, scaling Matrix dimensions over Native Hardware Multi-threads instantly without relying on external libraries like Python's PyTorch or TensorFlow.

## 🚀 Features

- **Interactive Dynamic I/O**: The framework automatically prompts you for your `.csv` dataset paths, calculates standard array features natively mathematically, and structurally scales the first Layer of your neural network inputs seamlessly avoiding all `NaN` Matrix crashes.
- **Object-Oriented Architecture**: Modular `Model`, `Layer`, `Optimizer`, and `Loss` interfaces.
- **OpenMP Multithreading**: C-block accelerated matrix arithmetic via parallelized CPU hardware multithreading (`#pragma omp parallel for`).
- **Memory Optimized**: O(n³) loop calculations are cache-line reordered natively allowing instant vector multiplications bounded to Native instruction APIs (AVX-2).
- **Advanced Optimizers**: Custom internal implementation of Adaptive Moment Estimation (`Adam`).
- **Model Storage**: Natively parse your trained Deep Learning multi-dimensional layers immediately back onto sequential binary disk files representing fully trained weights for quick inference reloading!

## ⚙️ Compilation

This engine runs at maximum speed utilizing strict performance execution flags. Ensure you have `g++` configured properly on your system.

```bash
g++ -O3 -march=native -mavx2 -fopenmp main.cpp math/*.cpp nn/*.cpp optim/*.cpp data/*.cpp trainer/*.cpp loss/*.cpp metrics/*.cpp scheduler/*.cpp model/*.cpp -o ml.exe
```

## 🧠 Using The Framework

Upload whatever dataset (e.g. `housing_prices.csv`) you wish to train into the folder. Run `./ml.exe` and the platform will specifically guide you to input your parameters securely via your standard Terminal stream:

```text
========================================
   UNIVERSAL C++ DEEP LEARNING ENGINE   
========================================
Enter path to CSV dataset (e.g., data/dataset.csv): data/housing_prices.csv
Enter number of target labels at the end of CSV: 1
Enter number of training epochs: 10
Enter filename to save trained model (e.g., model.bin): house_model.bin
```

Because of native structural parsing, the Matrix abstractions successfully deduce that `house_prices` has exactly `6` dimension columns and will structure and initialize its hidden Deep Learning framework perfectly scaled to your dataset requirements dynamically in the background!
