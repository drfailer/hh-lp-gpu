#ifndef TASK_FULLY_CONNECTED_LAYER_TASK_H
#define TASK_FULLY_CONNECTED_LAYER_TASK_H
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <memory>

struct LinearLayerTask : LayerTask {
    LinearLayerTask(std::string const &name, cudnnHandle_t cudnn_handle,
                    cublasHandle_t cublas_handle, size_t idx,
                    [[maybe_unused]] LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, idx) {
    }
    LinearLayerTask(cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle,
                    size_t idx, LayerDimentions const &dims)
        : LinearLayerTask("LinearLayerTask", cudnn_handle, cublas_handle,
                          idx, dims) {}

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) override {
        INFO_GRP("LinearLayerTask FWD", INFO_GRP_LAYER_TASK);
        auto &state = fwd_data->states[this->idx()];

        // save input (used for the backwards pass)
        state.input = fwd_data->input;

        CUDA_CHECK(memcpy_gpu_to_gpu(state.output, state.params.biases,
                                     state.dims.nb_nodes));
        cudaDeviceSynchronize();
        CUBLAS_CHECK(matvecmul(cublas(), false, state.dims.nb_nodes,
                               state.dims.nb_inputs, state.params.weights,
                               fwd_data->input, state.output));
        fwd_data->input = state.output;
        this->addResult(fwd_data);
    }

    void execute(std::shared_ptr<BwdData<ftype>> bwd_data) override {
        INFO_GRP("LinearLayerTask BWD", INFO_GRP_LAYER_TASK);
        auto &state = bwd_data->states[this->idx()];
        // Backward:
        // - grads_b = err, grads_w = err * fwd_inputT, err = err * w

        // TODO: for now we just copy but there might be more to do later
        CUDA_CHECK(memcpy_gpu_to_gpu(state.grads.biases, bwd_data->error,
                                     state.dims.nb_nodes));

        // w_grad = err * fwd_inputT
        CUBLAS_CHECK(matmul(cublas(), false, true, state.dims.nb_nodes,
                            state.dims.nb_inputs, 1, bwd_data->error,
                            state.input, state.grads.weights));

        // output_err = errT * weights
        CUBLAS_CHECK(matmul(cublas(), true, false, 1, state.dims.nb_inputs,
                            state.dims.nb_nodes, bwd_data->error,
                            state.params.weights,
                            state.error));

        bwd_data->error = state.error;
        this->addResult(bwd_data);
    }
};

#endif
