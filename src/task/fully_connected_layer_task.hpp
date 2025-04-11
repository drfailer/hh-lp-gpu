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
                            cublasHandle_t cublas_handle,
                            [[maybe_unused]] std::vector<int64_t> dims)
        : LayerTask(name), cudnn_handle_(cudnn_handle),
          cublas_handle_(cublas_handle) {}
    FullyConnectedLayerTask(cudnnHandle_t cudnn_handle,
                            cublasHandle_t cublas_handle,
                            std::vector<int64_t> dims)
        : FullyConnectedLayerTask("FullyConnectedLayerTask", cudnn_handle,
                                  cublas_handle, dims) {}

    ~FullyConnectedLayerTask() = default;

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) {
        INFO_GRP("FullyConnectedLayerTask FWD", INFO_GRP_LAYER_TASK);
        ftype alpha = 1, beta = 1;
        auto dims = fwd_data->layer->dims;

        CUDA_CHECK(memcpy_gpu_to_gpu(
            fwd_data->output_gpu, fwd_data->layer->biases_gpu, dims.nb_nodes));
        cudaDeviceSynchronize();
        CUBLAS_CHECK(cublasSgemv_v2(
            cublas_handle_, cublasOperation_t::CUBLAS_OP_N, dims.nb_nodes,
            dims.nb_inputs, &alpha, fwd_data->layer->weights_gpu,
            dims.nb_inputs, fwd_data->input_gpu, 1, &beta, fwd_data->output_gpu,
            1));
        fwd_data->layer++;
        this->addResult(fwd_data);
    }

    void execute(std::shared_ptr<BwdInputData<ftype>> bwd_data) {
        // TODO: execute the bwd graph
    }

  private:
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;
    cudnn_frontend::graph::Graph bwd_graph_;
};

#endif
