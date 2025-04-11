#ifndef TASK_FULLY_CONNECTED_LAYER_TASK_H
#define TASK_FULLY_CONNECTED_LAYER_TASK_H
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include "log.h/log.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <memory>
#include <vector>

struct FullyConnectedLayerTask : LayerTask {
    FullyConnectedLayerTask(std::string const &name, cudnnHandle_t cudnn_handle,
                            cublasHandle_t cublas_handle, size_t layer_idx,
                            [[maybe_unused]] LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, layer_idx),
          output_gpu_(nullptr) {
        size_t size = dims.nb_inputs * dims.nb_nodes;
        CUDA_CHECK(alloc_gpu(&output_gpu_, size));
    }
    FullyConnectedLayerTask(cudnnHandle_t cudnn_handle,
                            cublasHandle_t cublas_handle, size_t layer_idx,
                            LayerDimentions const &dims)
        : FullyConnectedLayerTask("FullyConnectedLayerTask", cudnn_handle,
                                  cublas_handle, layer_idx, dims) {}

    ~FullyConnectedLayerTask() { cudaFree(output_gpu_); }

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) override {
        INFO_GRP("FullyConnectedLayerTask FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 1;
        auto &layer = fwd_data->model.layers[this->layer_idx()];
        auto dims = layer.dims;

        CUDA_CHECK(
            memcpy_gpu_to_gpu(output_gpu_, layer.biases_gpu, dims.nb_nodes));
        cudaDeviceSynchronize();
        CUBLAS_CHECK(cublasSgemv_v2(
            cublas(), cublasOperation_t::CUBLAS_OP_N, dims.nb_nodes,
            dims.nb_inputs, &alpha, layer.weights_gpu, dims.nb_inputs,
            fwd_data->input_gpu, 1, &beta, output_gpu_, 1));
        fwd_data->input_gpu = output_gpu_;
        this->addResult(fwd_data);
    }

    void execute(std::shared_ptr<BwdData<ftype>> bwd_data) override {
        // TODO: execute the bwd graph
    }

  private:
    ftype *output_gpu_ = nullptr;
    cudnn_frontend::graph::Graph bwd_graph_;
};

#endif
