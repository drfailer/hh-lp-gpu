#ifndef MODEL_LOSS_LOSS_H
#define MODEL_LOSS_LOSS_H
#include "../data/tensor.hpp"

template <typename T> struct Loss {
    virtual Tensor<T> *fwd(Tensor<T> *model_output, Tensor<T> *ground_truth,
                           Tensor<T> *result) = 0;
    virtual Tensor<T> *bwd(Tensor<T> *model_output, Tensor<T> *ground_truth,
                           Tensor<T> *result) = 0;
};

#endif
