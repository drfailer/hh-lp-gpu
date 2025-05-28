#ifndef MODEL_DATA_LOSS_STATE
#define MODEL_DATA_LOSS_STATE
#include "tensor.hpp"

template <typename T> struct LossState {
    Tensor<T> *tensor = nullptr;

    LossState() = default;

    LossState(LossState const &) = delete;
    LossState const &operator=(LossState const &) = delete;

    ~LossState() {
        delete tensor;
    }
};

#endif
