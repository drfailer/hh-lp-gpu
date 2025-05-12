#ifndef MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#define MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#include "../../types.hpp"
#include "layer.hpp"
#include <cudnn_ops.h>
#include <log.h/log.h>
#include <cudnn.h>
#include <cudnn_graph.h>

struct SigmoidActivationLayer : Layer<ftype> {
    SigmoidActivationLayer(cudnnHandle_t cudnn_handle, int64_t size)
        : Layer(LayerDims{.inputs = size, .outputs = size}),
          cudnn_handle_(cudnn_handle) {
        // sigmoid activation tensor
        cudnnCreateActivationDescriptor(&sigmoid_);
        cudnnSetActivationDescriptor(sigmoid_, CUDNN_ACTIVATION_SIGMOID,
                                     CUDNN_NOT_PROPAGATE_NAN, 0);
        // input tensor
        cudnnCreateTensorDescriptor(&fwd_.input_tensor);
        cudnnSetTensor4dDescriptor(fwd_.input_tensor, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, 1, dims.inputs, 1);
        // output tensor
        cudnnCreateTensorDescriptor(&fwd_.output_tensor);
        cudnnSetTensor4dDescriptor(fwd_.output_tensor, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, 1, dims.inputs, 1);
        // input error tensor
        cudnnCreateTensorDescriptor(&bwd_.err_tensor);
        cudnnSetTensor4dDescriptor(bwd_.err_tensor, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, 1, dims.inputs, 1);
        // output error tensor
        cudnnCreateTensorDescriptor(&bwd_.output_tensor);
        cudnnSetTensor4dDescriptor(bwd_.output_tensor, CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT, 1, 1, dims.inputs, 1);
    }

    ~SigmoidActivationLayer() {
        cudnnDestroyTensorDescriptor(fwd_.input_tensor);
        cudnnDestroyTensorDescriptor(fwd_.output_tensor);
        cudnnDestroyTensorDescriptor(bwd_.err_tensor);
        cudnnDestroyTensorDescriptor(bwd_.output_tensor);
        cudnnDestroyActivationDescriptor(sigmoid_);
    }

    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradiants). Since this layer is an activation one,
     * there no need to allocate parameters and gradiants.
     */
    void init(LayerState<ftype> &state) override {
        state = create_layer_state<ftype>(this->dims, false, false);
    }

    ftype *fwd(LayerState<ftype> &state, ftype *input) override {
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        // save the input for the backwards pass
        state.input = input;

        cudnnActivationForward(cudnn_handle_, sigmoid_, &alpha,
                               fwd_.input_tensor, state.input, &beta,
                               fwd_.output_tensor, state.output);
        return state.output;
    }

    ftype *bwd(LayerState<ftype> &state, ftype *error) override {
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 0;

        cudnnActivationBackward(
            cudnn_handle_, sigmoid_, &alpha, fwd_.output_tensor, state.output,
            bwd_.err_tensor, error, fwd_.input_tensor, state.input, &beta,
            bwd_.output_tensor, state.error);
        return state.error;
    }

  private:
    struct {
        cudnnTensorDescriptor_t input_tensor;
        cudnnTensorDescriptor_t output_tensor;
    } fwd_;
    struct {
        cudnnTensorDescriptor_t err_tensor;
        cudnnTensorDescriptor_t output_tensor;
    } bwd_;
    cudnnActivationDescriptor_t sigmoid_;
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
