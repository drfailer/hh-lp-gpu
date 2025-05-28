#ifndef MODEL_LOSS_LOSS_H
#define MODEL_LOSS_LOSS_H
#include "../data/tensor.hpp"
#include "../data/loss_state.hpp"

template <typename T> struct Loss {
    virtual Tensor<T> *fwd(LossState<T> &state, Tensor<T> *model_output,
                           Tensor<T> *ground_truth) = 0;
    virtual Tensor<T> *bwd(LossState<T> &state, Tensor<T> *model_output,
                           Tensor<T> *ground_truth) = 0;
};

#endif
