#ifndef TASK_SIGMOID_ACTIVATION_TASK_H
#define TASK_SIGMOID_ACTIVATION_TASK_H
#include "../tools/gpu.hpp"
#include "layer_task.hpp"
#include <log.h/log.h>

struct SigmoidActivationTask : LayerTask {
    SigmoidActivationTask(std::string const &name, cudnnHandle_t cudnn_handle,
                          cublasHandle_t cublas_handle, size_t idx,
                          LayerDimentions const &dims)
        : LayerTask(name, cudnn_handle, cublas_handle, idx, dims) {
        build_fwd_graph(dims);
        build_bwd_graph(dims);
    }
    SigmoidActivationTask(cudnnHandle_t cudnn_handle,
                          cublasHandle_t cublas_handle, size_t idx,
                          LayerDimentions const &dims)
        : SigmoidActivationTask("SigmoidActivationTask", cudnn_handle,
                                cublas_handle, idx, dims) {}

    ~SigmoidActivationTask() {
        cudaFree(fwd_.workspace);
        cudaFree(bwd_.workspace);
    }

    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradiants). Since this layer is an activation one,
     * there no need to allocate parameters and gradiants.
     */
    void init(NetworkState<ftype> &state) override {
        state.layer_states[this->idx()] =
            layer_state_create_gpu<ftype>(this->dims(), {}, {});
    }

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        auto &state = data->states.layer_states[this->idx()];
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationTask FWD", INFO_GRP_LAYER_TASK);

        // save the input for the backwards pass
        state.input = data->input;

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {{fwd_.input_tensor, data->input},
                          {fwd_.output_tensor, state.output}};
        CUDNN_CHECK(fwd_.graph.execute(cudnn(), memory_map, fwd_.workspace));
        data->input = state.output;
        this->addResult(data);
    }

    void build_fwd_graph(LayerDimentions const &dims) {
        namespace fe = cudnn_frontend;
        auto &graph = fwd_.graph;

        fwd_.input_tensor =
            fwd_.graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("input")
                                  .set_dim({1, dims.nb_inputs, 1})
                                  .set_stride({dims.nb_inputs, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        fwd_.output_tensor = fwd_.graph.pointwise(
            fwd_.input_tensor,
            fe::graph::Pointwise_attributes()
                .set_name("sigmoid")
                .set_mode(fe::PointwiseMode_t::SIGMOID_FWD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        fwd_.output_tensor->set_output(true).set_data_type(
            fe::DataType_t::FLOAT);

        CUDNN_CHECK(graph.validate());
        CUDNN_CHECK(graph.build_operation_graph(cudnn()));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(cudnn()));
        CUDNN_CHECK(graph.build_plans(
            cudnn(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&fwd_.workspace, workspace_size));
    }

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        auto &state = data->states.layer_states[this->idx()];
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationTask BWD", INFO_GRP_LAYER_TASK);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {{bwd_.err_tensor, data->error},
                          {bwd_.fwd_input_tensor, state.input},
                          {bwd_.output_tensor, state.error}};
        CUDNN_CHECK(bwd_.graph.execute(cudnn(), memory_map, bwd_.workspace));
        data->error = state.error;
        this->addResult(data);
    }

    void build_bwd_graph(LayerDimentions const &dims) {
        namespace fe = cudnn_frontend;
        auto &graph = bwd_.graph;

        bwd_.err_tensor =
            bwd_.graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("input error")
                                  .set_dim({1, dims.nb_inputs, 1})
                                  .set_stride({dims.nb_inputs, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        bwd_.fwd_input_tensor =
            bwd_.graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("fwd input")
                                  .set_dim({1, dims.nb_inputs, 1})
                                  .set_stride({dims.nb_inputs, 1, 1})
                                  .set_data_type(fe::DataType_t::FLOAT));
        bwd_.output_tensor = bwd_.graph.pointwise(
            bwd_.err_tensor, bwd_.fwd_input_tensor,
            fe::graph::Pointwise_attributes()
                .set_name("sigmoid prime")
                .set_mode(fe::PointwiseMode_t::SIGMOID_BWD)
                .set_compute_data_type(fe::DataType_t::FLOAT));
        bwd_.output_tensor->set_output(true).set_data_type(
            fe::DataType_t::FLOAT);

        CUDNN_CHECK(graph.validate());
        CUDNN_CHECK(graph.build_operation_graph(cudnn()));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(cudnn()));
        CUDNN_CHECK(graph.build_plans(
            cudnn(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&bwd_.workspace, workspace_size));
    }

    void execute(std::shared_ptr<UpdateData<ftype>> data) override {
        this->addResult(data);
    }

  private:
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t input_tensor;
        tensor_attr_t output_tensor;
        float *workspace;
    } fwd_;
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t fwd_input_tensor;
        tensor_attr_t err_tensor;
        tensor_attr_t output_tensor;
        float *workspace;
    } bwd_;
};

#endif
