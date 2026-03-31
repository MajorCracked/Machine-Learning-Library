#pragma once
#include <vector>
#include <memory>
#include "../nn/layer.h"
#include "../optim/optimizer.h"


class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;

public:
    void add(Layer* l);
    void add(std::unique_ptr<Layer> l);
    Matrix forward(Matrix x);
    void backward(Matrix grad);
    void update(double lr);
    void step(Optimizer& opt);
    void save(const std::string& path);
    void load(const std::string& path);

    ~Model() = default;

};