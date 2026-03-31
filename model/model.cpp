#include "model.h"
#include "../optim/optimizer.h"
#include <fstream>
#include "../nn/dense.h"

void Model::add(Layer* l) {
    layers.emplace_back(l);
}

void Model::save(const std::string& path) {
    std::ofstream f(path, std::ios::binary);

    for(auto& l : layers) {
        Dense* d = dynamic_cast<Dense*>(l.get());
        if(d) d->save(f);
    }
}

void Model::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);

    for(auto& l : layers) {
        Dense* d = dynamic_cast<Dense*>(l.get());
        if(d) d->load(f);
    }
}

Matrix Model::forward(Matrix x) {
    for(auto &l : layers)
        x = l->forward(x);
    return x;
}

void Model::backward(Matrix grad) {
    for(int i=(int)layers.size()-1; i>=0; i--) {
        grad = layers[i]->backward(grad);
    }
}

void Model::update(double lr) {
    for(auto &l : layers)
        l->update(lr);
}

void Model::add(std::unique_ptr<Layer> l) {
    layers.push_back(std::move(l));
}

void Model::step(Optimizer& optim) {
    for (auto& layer : layers)
        optim.step(layer.get());   // ✅ pass pointer
}
