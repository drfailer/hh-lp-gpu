#ifndef MODEL_LOSS_LOSS_H
#define MODEL_LOSS_LOSS_H
#include "../data/cuda_data.hpp"
#include "../../tools/tensor/tensor.hpp"

struct LossFwdIn {
    tensor::Tensor<ftype> &x;
};

struct LossFwdOut {
    tensor::Tensor<ftype> &y;
};

struct LossBwdIn {
    tensor::Tensor<ftype> const &y_true;
    tensor::Tensor<ftype> const &y_pred;
};

struct LossBwdOut {
    tensor::Tensor<ftype> &dy;
};

template <typename T> struct Loss {
    virtual void fwd(CUDA cuda, LossFwdIn const &in, LossFwdOut const &out) = 0;
    virtual void bwd(CUDA cuda, LossBwdIn const &in, LossBwdOut const &out) = 0;
};

#endif
