#ifndef MODEL_LOSS_LOSS_H
#define MODEL_LOSS_LOSS_H
#include "../data/cuda_data.hpp"
#include "../data/loss_state.hpp"
#include "../../tools/tensor/tensor.hpp"

template <typename T> struct Loss {
    virtual tensor::Tensor<T> const &fwd(CUDA cuda_data, LossState<T> &state,
                                 tensor::Tensor<T> const &model_output,
                                 tensor::Tensor<T> const &ground_truth) = 0;
    virtual tensor::Tensor<T> const &bwd(CUDA cuda_data, LossState<T> &state,
                                 tensor::Tensor<T> const &model_output,
                                 tensor::Tensor<T> const &ground_truth) = 0;
};

#endif
