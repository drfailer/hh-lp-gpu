#ifndef MODEL_LOSS_LOSS_H
#define MODEL_LOSS_LOSS_H
#include "../data/cuda_data.hpp"
#include "../data/loss_state.hpp"
#include "../data/tensor.hpp"

template <typename T> struct Loss {
    virtual Tensor<T> *fwd(cuda_data_t cuda_data, LossState<T> &state,
                           Tensor<T> *model_output,
                           Tensor<T> *ground_truth) = 0;
    virtual Tensor<T> *bwd(cuda_data_t cuda_data, LossState<T> &state,
                           Tensor<T> *model_output,
                           Tensor<T> *ground_truth) = 0;
};

#endif
