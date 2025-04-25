#ifndef LAYERS_LINEAR_LAYER_H
#define LAYERS_LINEAR_LAYER_H
#include "../tools/gpu.hpp"
#include "../types.hpp"
#include "layer.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>

class LinearLayer : public Layer<ftype> {
  public:
    LinearLayer(cublasHandle_t cublas_handle, LayerDimentions const &dims)
        : Layer(dims), cublas_handle_(cublas_handle) {}

  public:
    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradiants).
     */
    void init(LayerState<ftype> &state) override {
        INFO_GRP("LinearLayer INIT", INFO_GRP_LAYER_TASK);
        auto params = parameters_create_gpu<ftype>(this->dims);
        auto grads = parameters_create_gpu<ftype>(this->dims);
        state = layer_state_create_gpu(this->dims, params, grads);
    }

    ftype *fwd(LayerState<ftype> &state, ftype *input) override {
        INFO_GRP("LinearLayer FWD", INFO_GRP_LAYER_TASK);

        // save input (used for the backwards pass)
        state.input = input;

        CUDA_CHECK(memcpy_gpu_to_gpu(state.output, state.params.biases,
                                     state.dims.nb_nodes));
        CUBLAS_CHECK(matvecmul(cublas_handle_, false, state.dims.nb_nodes,
                               state.dims.nb_inputs, 1.f, state.params.weights,
                               state.input, 1.f, state.output));
        return state.output;
    }

    ftype *bwd(LayerState<ftype> &state, ftype *error) override {
        INFO_GRP("LinearLayer BWD", INFO_GRP_LAYER_TASK);
        // Backward:
        // - grads_b = err, grads_w = err * update_inputT, err = err * w

        // TODO: for now we just copy but there might be more to do later
        CUDA_CHECK(
            memcpy_gpu_to_gpu(state.grads.biases, error, state.dims.nb_nodes));
        cudaDeviceSynchronize();

        // w_grad = err * update_inputT
        CUBLAS_CHECK(matmul(cublas_handle_, false, true, state.dims.nb_nodes,
                            state.dims.nb_inputs, 1, 1.f, error, state.input,
                            0.f, state.grads.weights));

        // output_err = errT * weights
        CUBLAS_CHECK(matmul(cublas_handle_, true, false, 1,
                            state.dims.nb_inputs, state.dims.nb_nodes, 1.f,
                            error, state.params.weights, 0.f, state.error));

        return state.error;
    }

  private:
    cublasHandle_t cublas_handle_ = nullptr;
};

#endif
