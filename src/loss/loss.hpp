#ifndef LOSS_LOSS_H
#define LOSS_LOSS_H
#include <cstdint>

template <typename T> struct Loss {
    virtual void init(int64_t size) = 0;
    virtual void fwd(T *model_output, T *ground_truth, T *result) = 0;
    virtual void bwd(T *model_output, T *ground_truth, T *result) = 0;
};

#endif
