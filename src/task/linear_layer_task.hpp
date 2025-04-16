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
        namespace fe = cudnn_frontend;
        auto &state = data->states[this->idx()];

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            weights_memory_map = {
                {update_.weights_tensor, state.params.weights},
                {update_.weights_scale_tensor, &data->learning_rate},
                {update_.weights_gradiant_tensor, state.grads.weights},
                {update_.weights_scaled_gradiant_tensor, state.grads.weights},
                {update_.weights_result_tensor, state.params.weights},
            };
        CUDNN_CHECK(update_.graph_weights.execute(cudnn(), weights_memory_map,
                                                  update_.workspace_weights));
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            biases_memory_map = {
                {update_.biases_tensor, state.params.biases},
                {update_.biases_scale_tensor, &data->learning_rate},
                {update_.biases_gradiant_tensor, state.grads.biases},
                {update_.biases_scaled_gradiant_tensor, state.grads.biases},
                {update_.biases_result_tensor, state.params.biases},
            };
        CUDNN_CHECK(update_.graph_biases.execute(cudnn(), biases_memory_map,
                                                 update_.workspace_biases));
        this->addResult(data);
    }

    void create_update_graph(LayerDimentions const &dims) {
        namespace fe = cudnn_frontend;
        auto &graph_weights = update_.graph_weights;
        auto &graph_biases = update_.graph_biases;
        auto data_type = fe::DataType_t::FLOAT;
        std::vector<int64_t> w_dims = {1, dims.nb_nodes, dims.nb_inputs},
                             w_strides = {dims.nb_nodes * dims.nb_inputs,
                                          dims.nb_inputs, 1};
        std::vector<int64_t> b_dims = {1, dims.nb_nodes, 1},
                             b_strides = {dims.nb_nodes, 1, 1};

        // WEIGHTS /////////////////////////////////////////////////////////////

        // tensor descriptor for the learning rate
        update_.weights_scale_tensor =
            graph_weights.tensor(fe::graph::Tensor_attributes()
                                     .set_name("learning rate")
                                     .set_dim({1, 1, 1})
                                     .set_stride({1, 1, 1})
                                     .set_data_type(data_type));
        // wieghts tensor
        update_.weights_tensor =
            graph_weights.tensor(fe::graph::Tensor_attributes()
                                     .set_name("weights")
                                     .set_dim(w_dims)
                                     .set_stride(w_strides)
                                     .set_data_type(data_type));
        // weights gradiants
        update_.weights_gradiant_tensor =
            graph_weights.tensor(fe::graph::Tensor_attributes()
                                     .set_name("weights gradiant")
                                     .set_dim(w_dims)
                                     .set_stride(w_strides)
                                     .set_data_type(data_type));
        // weights_grads *= learning_rate (or any scaling factor)
        update_.weights_scaled_gradiant_tensor =
            graph_weights
                .pointwise(update_.weights_scale_tensor,
                           update_.weights_gradiant_tensor,
                           fe::graph::Pointwise_attributes()
                               .set_name("multiply by learning rate")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(data_type));
        update_.weights_scaled_gradiant_tensor->set_data_type(data_type);
        // weights -= weights_grads
        update_.weights_result_tensor = graph_weights.pointwise(
            update_.weights_tensor, update_.weights_scaled_gradiant_tensor,
            fe::graph::Pointwise_attributes()
                .set_name("substract the scaled gradiant")
                .set_mode(fe::PointwiseMode_t::SUB)
                .set_compute_data_type(data_type));
        // result tensor
        update_.weights_result_tensor->set_output(true).set_data_type(
            data_type);

        CUDNN_CHECK(graph_weights.validate());
        CUDNN_CHECK(graph_weights.build_operation_graph(cudnn()));
        CUDNN_CHECK(graph_weights.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph_weights.check_support(cudnn()));
        CUDNN_CHECK(graph_weights.build_plans(
            cudnn(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph_weights.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&update_.workspace_biases, workspace_size));

        // BIASES //////////////////////////////////////////////////////////////

        // tensor descriptor for the learning rate
        update_.biases_scale_tensor =
            graph_biases.tensor(fe::graph::Tensor_attributes()
                                    .set_name("learning rate")
                                    .set_dim({1, 1, 1})
                                    .set_stride({1, 1, 1})
                                    .set_data_type(data_type));
        // biases tensor
        update_.biases_tensor =
            graph_biases.tensor(fe::graph::Tensor_attributes()
                                    .set_name("biases")
                                    .set_dim(b_dims)
                                    .set_stride(b_strides)
                                    .set_data_type(data_type));
        // biases gradiants
        update_.biases_gradiant_tensor =
            graph_weights.tensor(fe::graph::Tensor_attributes()
                                     .set_name("biases gradiant")
                                     .set_dim(b_dims)
                                     .set_stride(b_strides)
                                     .set_data_type(data_type));
        // biases_grads *= learning_rate (or any scaling factor)
        update_.biases_scaled_gradiant_tensor =
            graph_biases
                .pointwise(update_.biases_gradiant_tensor,
                           update_.biases_scale_tensor,
                           fe::graph::Pointwise_attributes()
                               .set_name("multiply by learning rate")
                               .set_mode(fe::PointwiseMode_t::MUL)
                               .set_compute_data_type(data_type));
        update_.biases_scaled_gradiant_tensor->set_data_type(data_type);
        // biases -= biases_grads
        update_.biases_result_tensor = graph_weights.pointwise(
            update_.biases_tensor, update_.biases_scaled_gradiant_tensor,
            fe::graph::Pointwise_attributes()
                .set_name("substract the scaled gradiant")
                .set_mode(fe::PointwiseMode_t::SUB)
                .set_compute_data_type(data_type));
        // result tensor
        update_.biases_result_tensor->set_output(true).set_data_type(data_type);

        CUDNN_CHECK(graph_biases.validate());
        CUDNN_CHECK(graph_biases.build_operation_graph(cudnn()));
        CUDNN_CHECK(graph_biases.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph_biases.check_support(cudnn()));
        CUDNN_CHECK(graph_biases.build_plans(
            cudnn(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        CUDNN_CHECK(graph_biases.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&update_.workspace_biases, workspace_size));
    }

  private:
    struct {
        cudnn_frontend::graph::Graph graph_biases;
        cudnn_frontend::graph::Graph graph_weights;
        tensor_attr_t weights_tensor;
        tensor_attr_t biases_tensor;
        tensor_attr_t weights_gradiant_tensor;
        tensor_attr_t biases_gradiant_tensor;
        tensor_attr_t weights_scaled_gradiant_tensor;
        tensor_attr_t biases_scaled_gradiant_tensor;
        tensor_attr_t weights_result_tensor;
        tensor_attr_t biases_result_tensor;
        tensor_attr_t weights_scale_tensor;
        tensor_attr_t biases_scale_tensor;
        ftype *workspace_biases = nullptr;
        ftype *workspace_weights = nullptr;
    } update_;
};

#endif
