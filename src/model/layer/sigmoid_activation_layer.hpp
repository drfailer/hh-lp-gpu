#ifndef MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#define MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

struct SigmoidActivationLayer : Layer<ftype> {
    SigmoidActivationLayer() : Layer({}) {
        // sigmoid activation tensor
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&sigmoid_));
        CUDNN_CHECK(cudnnSetActivationDescriptor(
            sigmoid_, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0));
    }

    ~SigmoidActivationLayer() override {
        cudnnDestroyActivationDescriptor(sigmoid_);
    }

    LayerIOShape io_shape(tensor::dims_t const &input_dims) const override {
        int inputs = input_dims[1] * input_dims[2] * input_dims[3];
        int outputs = inputs;
        int batch_size = input_dims[0];
        return LayerIOShape{
            .x = tensor::shape(batch_size, 1, inputs, 1),
            .y = tensor::shape(batch_size, 1, outputs, 1),
        };
    }

    void fwd(cuda_data_t cuda, fwd_data_t<ftype> const &data,
             tensor::Tensor<const ftype> const &x,
             tensor::Tensor<ftype> &y) override {
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationForward(cuda.cudnn_handle, sigmoid_, &alpha,
                                           x.desc(), x.data(), &beta, y.desc(),
                                           y.data()));
    }

    void bwd(cuda_data_t cuda, bwd_data_t<ftype> const &data,
             tensor::Tensor<const ftype> const &dy,
             tensor::Tensor<ftype> &dx) override {
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationBackward(
            cuda.cudnn_handle, sigmoid_, &alpha, data.y.desc(), data.y.data(),
            dy.desc(), dy.data(), data.x.desc(), data.x.data(), &beta,
            dx.desc(), dx.data()));
    }

  private:
    cudnnActivationDescriptor_t sigmoid_;
};

#endif
