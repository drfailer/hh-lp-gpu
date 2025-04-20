#ifndef TASK_FULLY_CONNECTED_LAYER_TASK_H
#define TASK_FULLY_CONNECTED_LAYER_TASK_H
#include "../tools/defer.hpp"
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include "log.h/log.h"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <memory>

#define TWO_GRAPHS

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
    void init(NetworkState<ftype> &state) override {
        auto params = parameters_create_gpu<ftype>(this->dims());
        auto grads = parameters_create_gpu<ftype>(this->dims());
        state.layer_states[this->idx()] =
            layer_state_create_gpu(this->dims(), params, grads);
    }

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        INFO_GRP("LinearLayerTask FWD", INFO_GRP_LAYER_TASK);
        auto &state = data->states.layer_states[this->idx()];

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
        auto &state = data->states.layer_states[this->idx()];
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
        auto &state = data->states.layer_states[this->idx()];
        ftype learning_rate_weights = data->learning_rate;
        ftype learning_rate_biases = data->learning_rate;
        MemoryMap mem_weights = {
            {update_weights_.scale_tensor, &learning_rate_weights}};
        MemoryMap mem_biases = {
            {update_biases_.scale_tensor, &learning_rate_biases}};

        map_update_graph_memory(mem_weights, update_weights_,
                                state.params.weights, state.grads.weights);
        map_update_graph_memory(mem_biases, update_biases_, state.params.biases,
                                state.grads.biases);

        // cudaDeviceSynchronize();
        CUDNN_CHECK(update_weights_.graph.execute(cudnn(), mem_weights,
                                                  update_weights_.workspace));
        CUDNN_CHECK(update_biases_.graph.execute(cudnn(), mem_biases,
                                                 update_biases_.workspace));
        cudaDeviceSynchronize();

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
        std::vector<int64_t> w_dims = {1, dims.nb_nodes, dims.nb_inputs},
                             w_strides = {dims.nb_nodes * dims.nb_inputs,
                                          dims.nb_inputs, 1};
        std::vector<int64_t> b_dims = {1, dims.nb_nodes, 1},
                             b_strides = {dims.nb_nodes, 1, 1};

        setup_update_graph(update_weights_, w_dims, w_strides);
        setup_update_graph(update_biases_, b_dims, b_strides);
    }

    void setup_update_graph(auto &data, std::vector<int64_t> const &dims,
                            std::vector<int64_t> const &strides) {
        namespace fe = cudnn_frontend;
        data.graph.set_io_data_type(fe::DataType_t::FLOAT)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

        // tenors attributes
        auto learning_rate_attributes = fe::graph::Tensor_attributes()
                                            .set_name("learning rate")
                                            .set_dim({1, 1, 1})
                                            .set_stride({1, 1, 1});
        auto param_attributes = fe::graph::Tensor_attributes()
                                    .set_name("parameter")
                                    .set_dim(dims)
                                    .set_stride(strides);
        auto grad_attributes = fe::graph::Tensor_attributes()
                                   .set_name("parameter gradiant")
                                   .set_dim(dims)
                                   .set_stride(strides);
        auto scale_attributes = fe::graph::Pointwise_attributes()
                                    .set_name("learning_rate * gradiant")
                                    .set_mode(fe::PointwiseMode_t::MUL);
        auto substract_attributes =
            fe::graph::Pointwise_attributes()
                .set_name("parameters - learning_rate * gradiant")
                .set_mode(fe::PointwiseMode_t::SUB);

        // graph inputs:

        data.tensor = data.graph.tensor(param_attributes);
        data.gradiant_tensor = data.graph.tensor(grad_attributes);
        data.scale_tensor = data.graph.tensor(learning_rate_attributes);

        // operations:

        // scaled_gradiant = gradiants * learning_rate
        data.scaled_gradiant_tensor = data.graph.pointwise(
            data.scale_tensor, data.gradiant_tensor, scale_attributes);
        // parameters -= scaled_gradiant
        data.result_tensor = data.graph.pointwise(
            data.tensor, data.scaled_gradiant_tensor, substract_attributes);

        // result:

        data.result_tensor->set_output(true);

        CUDNN_CHECK(data.graph.validate());
        CUDNN_CHECK(data.graph.build(cudnn(), {fe::HeurMode_t::A}));

        int64_t workspace_size;
        CUDNN_CHECK(data.graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&data.workspace, workspace_size));
    }

  private:
    struct LinearLayerUpdateData {
        cudnn_frontend::graph::Graph graph;
        ftype *workspace = nullptr;
        tensor_attr_t tensor;
        tensor_attr_t gradiant_tensor;
        tensor_attr_t scaled_gradiant_tensor;
        tensor_attr_t result_tensor;
        tensor_attr_t scale_tensor;
    };
    LinearLayerUpdateData update_weights_;
    LinearLayerUpdateData update_biases_;
};

#endif
