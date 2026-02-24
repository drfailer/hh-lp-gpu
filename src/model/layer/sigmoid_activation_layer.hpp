#ifndef MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#define MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn.h>
#include <cudnn_graph.h>
#include <cudnn_ops.h>

struct SigmoidActivationLayer : Layer {
    SigmoidActivationLayer(tensor::dtype_t dtype = CUDNN_DATA_TYPE)
        : Layer({}, dtype) {
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

    void fwd(CUDA cuda, LayerFwdIn const &in, LayerFwdOut const &out) override {
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationForward(cuda.cudnn_handle, sigmoid_, &alpha,
                                           in.x.desc(), in.x.data(), &beta,
                                           out.y.desc(), out.y.data()));
    }

    void bwd(CUDA cuda, LayerBwdIn const &in, LayerBwdOut const &out) override {
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        CUDNN_CHECK(cudnnActivationBackward(
            cuda.cudnn_handle, sigmoid_, &alpha, in.y.desc(), in.y.data(),
            in.dy.desc(), in.dy.data(), in.x.desc(), in.x.data(), &beta,
            out.dx.desc(), out.dx.data()));
    }

  private:
    cudnnActivationDescriptor_t sigmoid_;
};

#endif
