#ifndef GRAPH_NETWORK_GRAPH_H
#define GRAPH_NETWORK_GRAPH_H
#include "../state/pipeline_state_manager.hpp"
#include "../task/layer_task.hpp"
#include "../task/loss/loss_task.hpp"
#include <hedgehog/hedgehog.h>

#define NetworkGraphIn InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphOut InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphIO 2, NetworkGraphIn, NetworkGraphOut

class NetworkGraph : public hh::Graph<NetworkGraphIO> {
  public:
    NetworkGraph() : hh::Graph<NetworkGraphIO>() {
        pipeline_state = std::make_shared<PipelineStateManager>(
            std::make_shared<PipelineState>());

        this->inputs(pipeline_state);
        this->outputs(pipeline_state);
    }

    void add_layer(std::shared_ptr<LayerTask> layer) {
        layers.emplace_back(layer);
    }

    void set_loss(std::shared_ptr<LossTask> loss_task) {
        this->loss_task = loss_task;
    }

    void build() {
        this->edges(pipeline_state, layers.front());
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            this->edges(layers[i], layers[i + 1]);
        }
        this->edges(layers.back(), pipeline_state);
    }

    void init_network_state(NetworkState<ftype> &state) {
        state.layer_states = std::vector<LayerState<ftype>>(layers.size());
        for (auto layer : layers) {
            layer->init(state);
        }
        loss_task->init(state);
    }

    void destroy_network_state(NetworkState<ftype> &state) {
        for (auto &layer_state : state.layer_states) {
            parameters_destroy_gpu(layer_state.params);
            parameters_destroy_gpu(layer_state.grads);
            layer_state_destroy_gpu(layer_state);
        }
        cudaFree(state.loss_output);
    }

  private:
    std::shared_ptr<PipelineStateManager> pipeline_state = nullptr;
    std::shared_ptr<LossTask> loss_task = nullptr;
    std::vector<std::shared_ptr<LayerTask>> layers = {};
};

#endif
