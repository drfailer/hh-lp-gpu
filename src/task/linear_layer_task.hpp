#ifndef TASK_FULLY_CONNECTED_LAYER_TASK_H
#define TASK_FULLY_CONNECTED_LAYER_TASK_H
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <memory>
#include <ostream>

struct LinearLayerTask : LayerTask {
    LinearLayerTask(std::string const &name, cudnnHandle_t cudnn_handle,
                    cublasHandle_t cublas_handle, size_t idx,
                    LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, idx, dims) {
        create_update_graph(dims);
    }
    LinearLayerTask(cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle,
                    size_t idx, LayerDimentions const &dims)
        : LinearLayerTask("LinearLayerTask", cudnn_handle, cublas_handle, idx,
                          dims) {}

    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradiants).
     */
    void execute(std::shared_ptr<InitData<ftype>> data) override {
        auto &state = data->states[this->idx()];
        auto params = parameters_create_gpu<ftype>(this->dims());
        auto grads = parameters_create_gpu<ftype>(this->dims());
        state = layer_state_create_gpu(this->dims(), params, grads);
        this->addResult(data);
    }

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        INFO_GRP("LinearLayerTask FWD", INFO_GRP_LAYER_TASK);
        auto &state = data->states[this->idx()];

        // save input (used for the backwards pass)
        state.input = data->input;

        CUDA_CHECK(memcpy_gpu_to_gpu(state.output, state.params.biases,
                                     state.dims.nb_nodes));
        cudaDeviceSynchronize();
        CUBLAS_CHECK(matvecmul(cublas(), false, state.dims.nb_nodes,
                               state.dims.nb_inputs, state.params.weights,
                               data->input, state.output));
        data->input = state.output;
        this->addResult(data);
    }

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        INFO_GRP("LinearLayerTask BWD", INFO_GRP_LAYER_TASK);
        auto &state = data->states[this->idx()];
        // Backward:
        // - grads_b = err, grads_w = err * update_inputT, err = err * w

        // TODO: for now we just copy but there might be more to do later
        CUDA_CHECK(memcpy_gpu_to_gpu(state.grads.biases, data->error,
                                     state.dims.nb_nodes));

        // w_grad = err * update_inputT
        CUBLAS_CHECK(matmul(cublas(), false, true, state.dims.nb_nodes,
                            state.dims.nb_inputs, 1, data->error, state.input,
                            state.grads.weights));

        // output_err = errT * weights
        CUBLAS_CHECK(matmul(cublas(), true, false, 1, state.dims.nb_inputs,
                            state.dims.nb_nodes, data->error,
                            state.params.weights, state.error));

        data->error = state.error;
        this->addResult(data);
    }

    void execute(std::shared_ptr<UpdateData<ftype>> data) override {
        INFO_GRP("LinearLayerTask Update", INFO_GRP_LAYER_TASK);
        auto &state = data->states[this->idx()];
        namespace fe = cudnn_frontend;
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            mem_map = {{scale_tensor_, &data->learning_rate}};
        map_update_graph_memory(mem_map, update_weights_, state.params.weights,
                                state.grads.weights);
        map_update_graph_memory(mem_map, update_biases_, state.params.biases,
                                state.grads.biases);
        CUDNN_CHECK(graph_.execute(cudnn(), mem_map, workspace));
        this->addResult(data);
    }

    void map_update_graph_memory(auto &mem_map, auto &update_data,
                                 ftype *parameters, ftype *gradiants) {
        mem_map.insert({update_data.tensor, parameters});
        mem_map.insert({update_data.gradiant_tensor, gradiants});
        mem_map.insert({update_data.scaled_gradiant_tensor, gradiants});
        mem_map.insert({update_data.result_tensor, parameters});
    }

    void create_update_graph(LayerDimentions const &dims) {
        INFO("create_update_graph");
        namespace fe = cudnn_frontend;
        std::vector<int64_t> w_dims = {1, dims.nb_nodes, dims.nb_inputs},
                             w_strides = {dims.nb_nodes * dims.nb_inputs,
                                          dims.nb_inputs, 1};
        std::vector<int64_t> b_dims = {1, dims.nb_nodes, 1},
                             b_strides = {dims.nb_nodes, 1, 1};

        graph_.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        // tenors attributes
        auto learning_attributes = fe::graph::Tensor_attributes()
                                         .set_name("learning rate")
                                         .set_dim({1, 1, 1})
                                         .set_stride({1, 1, 1});
        auto weights_attributes = fe::graph::Tensor_attributes()
                                      .set_name("weights")
                                      .set_dim(w_dims)
                                      .set_stride(w_strides);
        auto biases_attributes = fe::graph::Tensor_attributes()
                                     .set_name("biases")
                                     .set_dim(b_dims)
                                     .set_stride(b_strides);
        auto scale_attributes = fe::graph::Pointwise_attributes()
                                    .set_name("learning_rate * gradiant")
                                    .set_mode(fe::PointwiseMode_t::MUL);
        auto substract_attributes =
            fe::graph::Pointwise_attributes()
                .set_name("parameters - learning_rate * gradiant")
                .set_mode(fe::PointwiseMode_t::SUB);

        // graph inputs:

        scale_tensor_ = graph_.tensor(learning_attributes);
        update_weights_.tensor = graph_.tensor(weights_attributes);
        update_biases_.tensor = graph_.tensor(biases_attributes);
        update_weights_.gradiant_tensor = graph_.tensor(weights_attributes);
        update_biases_.gradiant_tensor = graph_.tensor(biases_attributes);

        // operations:

        // scaled_gradiant = gradiants * learning_rate
        update_weights_.scaled_gradiant_tensor = graph_.pointwise(
            scale_tensor_, update_weights_.gradiant_tensor, scale_attributes);
        update_biases_.scaled_gradiant_tensor = graph_.pointwise(
            scale_tensor_, update_biases_.gradiant_tensor, scale_attributes);
        // parameters -= scaled_gradiant
        update_weights_.result_tensor = graph_.pointwise(
            update_weights_.tensor, update_weights_.scaled_gradiant_tensor,
            substract_attributes);
        update_biases_.result_tensor = graph_.pointwise(
            update_biases_.tensor, update_biases_.scaled_gradiant_tensor,
            substract_attributes);

        // result:

        update_weights_.result_tensor->set_output(true);
        update_biases_.result_tensor->set_output(true);

        CUDNN_CHECK(graph_.validate());
        CUDNN_CHECK(graph_.build(cudnn(), {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(graph_.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&workspace, workspace_size));
    }

  private:
    cudnn_frontend::graph::Graph graph_;
    ftype *workspace = nullptr;
    tensor_attr_t scale_tensor_;
    struct LinearLayerUpdateData {
        tensor_attr_t tensor;
        tensor_attr_t gradiant_tensor;
        tensor_attr_t scaled_gradiant_tensor;
        tensor_attr_t result_tensor;
    };
    LinearLayerUpdateData update_weights_;
    LinearLayerUpdateData update_biases_;
};

#endif
