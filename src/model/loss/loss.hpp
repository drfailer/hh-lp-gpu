#ifndef MODEL_LOSS_LOSS_H
#define MODEL_LOSS_LOSS_H
#include "../data/cuda_data.hpp"
#include "../../tools/tensor/tensors.hpp"

struct LossFwdIn {
    tensor::Tensor &x;
};

struct LossFwdOut {
    tensor::Tensor &y;
};

struct LossBwdIn {
    tensor::Tensor const &y_true;
    tensor::Tensor const &y_pred;
};

struct LossBwdOut {
    tensor::Tensor &dy;
};

struct Loss {
    virtual void fwd(CUDA cuda, LossFwdIn const &in, LossFwdOut const &out) = 0;
    virtual void bwd(CUDA cuda, LossBwdIn const &in, LossBwdOut const &out) = 0;
};

#endif
