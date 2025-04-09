#ifndef TASK_FULLY_CONNECTED_LAYER_TASK_H
#define TASK_FULLY_CONNECTED_LAYER_TASK_H
#include "../utils.hpp"
#include "layer_task.hpp"
#include <cudnn.h>
#include <cudnn_frontend/graph_interface.h>
#include <cudnn_graph.h>
#include <memory>
#include <vector>

struct FullyConnectedLayerTask : LayerTask {
    FullyConnectedLayerTask(cudnnHandle_t *handle, std::vector<int64_t> dims)
        : LayerTask("FullyConnectedLayerTask"), handle_(handle) {
        build_fwd_graph(dims);
        build_bwd_graph(dims);
    }

    ~FullyConnectedLayerTask() { cudaFree(fwd_.workspace); }

    void execute(std::shared_ptr<FwdInputData<ftype>> fwd_data) {
        std::unordered_map<
            std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void *>
            memory_map = {
                {fwd_.input_tensor, fwd_data->layer.weights_cpu},
                {fwd_.weights_tensor, fwd_data->layer.weights_gpu},
                {fwd_.bias_tensor, fwd_data->layer.biases_gpu},
                {fwd_.z_tensor, fwd_data->z_gpu}, // TODO
                {fwd_.act_tensor, fwd_data->act_gpu},
            };
        CUDNN_CHECK(fwd_.graph.execute(*handle_, memory_map, fwd_.workspace));
        this->addResult(std::make_shared<FwdOutputData<ftype>>(fwd_data->z_gpu,
                    fwd_data->act_gpu));
    }

    void execute(std::shared_ptr<BwdInputData<ftype>> bwd_data) {
        // TODO: execute the bwd graph
    }

    /*
     * z = weights*fwd.input + biases
     * a = act(z)
     */
    void build_fwd_graph(std::vector<int64_t> dims) {
        namespace fe = cudnn_frontend;

        auto &graph = fwd_.graph;
        int64_t nb_inputs = dims[1];
        int64_t nb_nodes = dims[2];

        fwd_.weights_tensor =
            graph.tensor(fe::graph::Tensor_attributes()
                             .set_name("weights")
                             .set_dim(dims)
                             .set_stride({nb_inputs * nb_nodes, nb_nodes, 1})
                             .set_data_type(COMPUTE_DATA_TYPE));
        fwd_.bias_tensor = graph.tensor(fe::graph::Tensor_attributes()
                                            .set_name("biases")
                                            .set_dim({1, 1, nb_nodes})
                                            .set_stride({nb_nodes, 1, 1})
                                            .set_data_type(COMPUTE_DATA_TYPE));
        fwd_.input_tensor = graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("input")
                                             .set_dim({1, 1, nb_inputs})
                                             .set_stride({nb_inputs, 1, 1})
                                             .set_data_type(COMPUTE_DATA_TYPE));

        fwd_.z_tensor =
            graph.matmul(fwd_.weights_tensor, fwd_.input_tensor,
                         fe::graph::Matmul_attributes()
                             .set_name("z_matmul")
                             .set_compute_data_type(COMPUTE_DATA_TYPE));
        // TODO: the activation function should be configurable
        fwd_.act_tensor = graph.pointwise(
            fwd_.z_tensor, fe::graph::Pointwise_attributes()
                               .set_name("activation")
                               .set_mode(fe::PointwiseMode_t::SIGMOID_FWD)
                               .set_compute_data_type(COMPUTE_DATA_TYPE));
        fwd_.act_tensor->set_output(true).set_data_type(COMPUTE_DATA_TYPE);

        CUDNN_CHECK(graph.validate());
        CUDNN_CHECK(graph.build_operation_graph(*handle_));
        CUDNN_CHECK(graph.create_execution_plans({fe::HeurMode_t::A}));
        CUDNN_CHECK(graph.check_support(*handle_));
        CUDNN_CHECK(graph.build_plans(
            *handle_, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE));

        int64_t workspace_size;
        CUDNN_CHECK(graph.get_workspace_size(workspace_size));
        fwd_.workspace = nullptr;
        cudaMalloc((void **)(&fwd_.workspace),
                   workspace_size * sizeof(*fwd_.workspace));
    }

    void build_bwd_graph(std::vector<int64_t> dims) {
        // TODO
    }

  private:
    cudnnHandle_t *handle_;
    struct {
        cudnn_frontend::graph::Graph graph;
        tensor_attr_t weights_tensor;
        tensor_attr_t bias_tensor;
        tensor_attr_t input_tensor;
        tensor_attr_t act_tensor;
        tensor_attr_t z_tensor;
        float *workspace;
        ftype *z_gpu;
        ftype *output_gpu;
    } fwd_;
    cudnn_frontend::graph::Graph bwd_graph_;
};

#endif
