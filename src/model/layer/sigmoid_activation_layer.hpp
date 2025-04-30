#ifndef MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#define MODEL_LAYER_SIGMOID_ACTIVATION_LAYER_H
#include "../../tools/gpu.hpp"
#include "../../types.hpp"
#include "layer.hpp"
#include <log.h/log.h>

struct SigmoidActivationLayer : Layer<ftype> {
    SigmoidActivationLayer(cudnnHandle_t cudnn_handle,
                           int64_t size)
        : Layer(LayerDimentions{.nb_nodes = size, .nb_inputs = size,
                .kernel_size = 1}), cudnn_handle_(cudnn_handle) {
        build_fwd_graph(dims);
        build_bwd_graph(dims);
    }

    ~SigmoidActivationLayer() {
        cudaFree(fwd_.workspace);
        cudaFree(bwd_.workspace);
    }

    /*
     * Allocates memory for a layer state (output memory for the fwd pass, bwd
     * pass, parameters and gradiants). Since this layer is an activation one,
     * there no need to allocate parameters and gradiants.
     */
    void init(LayerState<ftype> &state) override {
        state = layer_state_create_gpu<ftype>(this->dims, {}, {});
    }

    ftype *fwd(LayerState<ftype> &state, ftype *input) override {
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationLayer FWD", INFO_GRP_LAYER_TASK);

        // save the input for the backwards pass
        state.input = input;

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {{fwd_.input_tensor, state.input},
                          {fwd_.output_tensor, state.output}};
        CUDNN_CHECK(
            fwd_.graph.execute(cudnn_handle_, memory_map, fwd_.workspace));
        return state.output;
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
        CUDNN_CHECK(graph.build_operation_graph(cudnn_handle_));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(cudnn_handle_));
        CUDNN_CHECK(graph.build_plans(
            cudnn_handle_, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&fwd_.workspace, workspace_size));
    }

    ftype *bwd(LayerState<ftype> &state, ftype *error) override {
        namespace fe = cudnn_frontend;
        INFO_GRP("SigmoidActivationLayer BWD", INFO_GRP_LAYER_TASK);

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>,
                           void *>
            memory_map = {{bwd_.err_tensor, error},
                          {bwd_.fwd_input_tensor, state.input},
                          {bwd_.output_tensor, state.error}};
        CUDNN_CHECK(
            bwd_.graph.execute(cudnn_handle_, memory_map, bwd_.workspace));
        return state.error;
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
        CUDNN_CHECK(graph.build_operation_graph(cudnn_handle_));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(cudnn_handle_));
        CUDNN_CHECK(graph.build_plans(
            cudnn_handle_, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        CUDA_CHECK(alloc_gpu(&bwd_.workspace, workspace_size));
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
    cudnnHandle_t cudnn_handle_ = nullptr;
};

#endif
