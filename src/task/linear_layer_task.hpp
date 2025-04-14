#ifndef TASK_FULLY_CONNECTED_LAYER_TASK_H
#define TASK_FULLY_CONNECTED_LAYER_TASK_H
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <memory>
#include <vector>

struct LinearLayerTask : LayerTask {
    LinearLayerTask(std::string const &name, cudnnHandle_t cudnn_handle,
                            cublasHandle_t cublas_handle, size_t layer_idx,
                            [[maybe_unused]] LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, layer_idx),
          output_gpu_(nullptr) {
        size_t size = dims.nb_inputs * dims.nb_nodes;
        CUDA_CHECK(alloc_gpu(&output_gpu_, size));
    }
    LinearLayerTask(cudnnHandle_t cudnn_handle,
                            cublasHandle_t cublas_handle, size_t layer_idx,
                            LayerDimentions const &dims)
        : LinearLayerTask("LinearLayerTask", cudnn_handle,
                                  cublas_handle, layer_idx, dims) {}

    ~LinearLayerTask() { cudaFree(output_gpu_); }

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) override {
        INFO_GRP("LinearLayerTask FWD", INFO_GRP_LAYER_TASK);
        auto &layer = fwd_data->model.layers[this->layer_idx()];
        auto dims = layer.dims;

        CUDA_CHECK(
            memcpy_gpu_to_gpu(output_gpu_, layer.biases_gpu, dims.nb_nodes));
        cudaDeviceSynchronize();
        CUBLAS_CHECK(matvecmul(cublas(), false, dims.nb_nodes, dims.nb_inputs,
                               layer.weights_gpu, fwd_data->input_gpu,
                               output_gpu_));
        fwd_data->input_gpu = output_gpu_;
        this->addResult(fwd_data);
    }

    void execute(std::shared_ptr<BwdData<ftype>> bwd_data) override {
        // TODO: execute the bwd graph
        // Backward:
        // - grads_b = err, grads_w = matmul(fwd_inputT, err), err = matmul(err, w)
    }

  private:
    ftype *output_gpu_ = nullptr;
    cudnn_frontend::graph::Graph bwd_graph_;
};

#endif
