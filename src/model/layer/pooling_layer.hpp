#ifndef MODEL_LAYER_POOLING_LAYER
#define MODEL_LAYER_POOLING_LAYER
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn_ops.h>

struct PoolingLayer : Layer<ftype> {
    cudnnPoolingDescriptor_t pooling_descriptor;
    cudnnTensorDescriptor_t input_descriptor;

    PoolingLayer(cudnnPoolingMode_t mode, int width, int height,
                 int horizontal_padding = 0, int vertical_padding = 0)
        : Layer<ftype>({.kernel_width = width, .kernel_height = height}) {
        int horizontal_stride = width;
        int vertical_stride = height;

        CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pooling_descriptor));
        // TODO: we might want to use the Nd version at some point
        // cudnnSetPoolingNdDescriptor(
        //     pooling_descriptor, mode, CUDNN_NOT_PROPAGATE_NAN,
        //     window_dims.size(),
        //     window_dims.data(), padding.data(), strides.data());
        CUDNN_CHECK(cudnnSetPooling2dDescriptor(
            pooling_descriptor, mode, CUDNN_NOT_PROPAGATE_NAN, height, width,
            vertical_padding, horizontal_padding, vertical_stride,
            horizontal_stride));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
    }

    ~PoolingLayer() override {
        CUDNN_CHECK(cudnnDestroyPoolingDescriptor(pooling_descriptor));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptor));
    }

    LayerIOShape io_shape(tensor::dims_t const &input_dims) const override {
        tensor::dims_t output_dims;
        cudnnSetTensorNdDescriptorEx(input_descriptor, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_TYPE, input_dims.size(),
                                     input_dims.data());
        cudnnGetPoolingNdForwardOutputDim(pooling_descriptor, input_descriptor,
                                          output_dims.size(),
                                          output_dims.data());
        // here we are stuck and we need the full input
        return {
            .x = tensor::shape(input_dims),
            .y = tensor::shape(output_dims),
        };
    }

    void init_fwd(CUDA cuda, InitFwdData const &data) override {
        tensor::dims_t input_dims = data.x.dims();
        cudnnSetTensorNdDescriptorEx(input_descriptor, CUDNN_TENSOR_NCHW,
                                     CUDNN_DATA_TYPE, input_dims.size(),
                                     input_dims.data());
    }

    void fwd(CUDA cuda, FwdIn const &in, FwdOut const &out) override {
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingForward(cuda.cudnn_handle, pooling_descriptor,
                                        &alpha, in.x.desc(), in.x.data(), &beta,
                                        out.y.desc(), out.y.data()));
    }

    void bwd(CUDA cuda, BwdIn const &in, BwdOut const &out) override {
        auto error_descriptor = in.dy.desc();
        auto error_data = in.dy.data();
        ftype alpha = 1;
        ftype beta = 0;

        CUDNN_CHECK(cudnnPoolingBackward(
            cuda.cudnn_handle, pooling_descriptor, &alpha, in.y.desc(),
            in.y.data(), error_descriptor, error_data, in.x.desc(), in.x.data(),
            &beta, out.dx.desc(), out.dx.data()));
    }
};

#endif
