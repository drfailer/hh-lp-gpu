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
          fwd_output_gpu_(nullptr), dims(dims) {
        size_t size = dims.nb_inputs * dims.nb_nodes;
        CUDA_CHECK(alloc_gpu(&fwd_output_gpu_, dims.nb_nodes));
        CUDA_CHECK(alloc_gpu(&weights_gradiant_gpu_, size));
        CUDA_CHECK(alloc_gpu(&biases_gradiant_gpu_, dims.nb_nodes));
        CUDA_CHECK(alloc_gpu(&bwd_output_gpu_, dims.nb_inputs));
    }
    LinearLayerTask(cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle,
                    size_t layer_idx, LayerDimentions const &dims)
        : LinearLayerTask("LinearLayerTask", cudnn_handle, cublas_handle,
                          layer_idx, dims) {}

    ~LinearLayerTask() {
        cudaFree(fwd_output_gpu_);
        cudaFree(weights_gradiant_gpu_);
        cudaFree(biases_gradiant_gpu_);
        cudaFree(bwd_output_gpu_);
    }

    void execute(std::shared_ptr<FwdData<ftype>> fwd_data) override {
        INFO_GRP("LinearLayerTask FWD", INFO_GRP_LAYER_TASK);
        auto &layer = fwd_data->model.layers[this->layer_idx()];
        auto dims = layer.dims;

        // save input
        fwd_input_gpu_ = fwd_data->input_gpu;

        CUDA_CHECK(memcpy_gpu_to_gpu(fwd_output_gpu_, layer.biases_gpu,
                                     dims.nb_nodes));
        cudaDeviceSynchronize();
        CUBLAS_CHECK(matvecmul(cublas(), false, dims.nb_nodes, dims.nb_inputs,
                               layer.weights_gpu, fwd_data->input_gpu,
                               fwd_output_gpu_));
        fwd_data->input_gpu = fwd_output_gpu_;
        this->addResult(fwd_data);
    }

    void execute(std::shared_ptr<BwdData<ftype>> bwd_data) override {
        // TODO: execute the bwd graph
        // Backward:
        // - grads_b = err, grads_w = matmul(fwd_inputT, err), err = matmul(err,
        // w)
        ftype *weights_gpu = bwd_data->model.layers[layer_idx()].weights_gpu;

        // TODO: the gradiants should be reset after the optimization (gemm adds
        // to the result vector)

        // TODO: for now we just copy but there might be more to do later
        CUDA_CHECK(memcpy_gpu_to_gpu(biases_gradiant_gpu_, bwd_data->err_gpu,
                                     dims.nb_nodes));

        // w_grad = err * fwd_inputT
        CUBLAS_CHECK(matmul(cublas(), false, true, dims.nb_nodes,
                            dims.nb_inputs, 1, bwd_data->err_gpu,
                            fwd_input_gpu_, weights_gradiant_gpu_));

        // output_err = errT * weights
        CUBLAS_CHECK(matmul(cublas(), true, false, 1, dims.nb_inputs,
                            dims.nb_nodes, bwd_data->err_gpu, weights_gpu,
                            bwd_output_gpu_));

        bwd_data->err_gpu = bwd_output_gpu_;
        this->addResult(bwd_data);
    }

  private:
    ftype *fwd_input_gpu_ = nullptr;
    ftype *fwd_output_gpu_ = nullptr;
    ftype *bwd_output_gpu_ = nullptr;
    ftype *weights_gradiant_gpu_ = nullptr;
    ftype *biases_gradiant_gpu_ = nullptr;
    LayerDimentions dims;
    cudnn_frontend::graph::Graph bwd_graph_;
};

#endif
