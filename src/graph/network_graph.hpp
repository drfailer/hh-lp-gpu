#ifndef GRAPH_NETWORK_GRAPH_H
#define GRAPH_NETWORK_GRAPH_H
#include "../state/inference_state.hpp"
#include "../state/inference_state_manager.hpp"
#include "../state/training_state.hpp"
#include "../task/layer_task.hpp"
#include "../task/loss/loss_task.hpp"
#include <hedgehog/hedgehog.h>

#define NetworkGraphIn InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphOut InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphIO 2, NetworkGraphIn, NetworkGraphOut

class NetworkGraph : public hh::Graph<NetworkGraphIO> {
  public:
    NetworkGraph() : hh::Graph<NetworkGraphIO>() {
        inference_state = std::make_shared<InferenceStateManager>(
            std::make_shared<InferenceState>());
        training_state = std::make_shared<hh::StateManager<TrainingStateIO>>(
            std::make_shared<TrainingState>());

        this->inputs(inference_state);
        // this->inputs(training_state);
        this->outputs(inference_state);
        // this->outputs(training_state);
    }

    void add_layer(std::shared_ptr<LayerTask> layer) {
        layers.emplace_back(layer);
    }

    void set_loss(std::shared_ptr<LossTask> loss_task) {
        this->loss_task = loss_task;
    }

    void build() {
        this->edges(inference_state, layers.front());
        // this->edges(training_state, layers.front());
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            this->edges(layers[i], layers[i + 1]);
        }
        this->edges(layers.back(), inference_state);
        // this->edges(layers.back(), training_state);
    }

    void init_network_state(NetworkState<ftype> &state) {
        state.layer_states = std::vector<LayerState<ftype>>(layers.size());
        for (auto layer : layers) {
            layer->init(state);
        }
        loss_task->init(state);
    }

  private:
    std::shared_ptr<InferenceStateManager> inference_state =
        nullptr;
    std::shared_ptr<hh::StateManager<TrainingStateIO>> training_state = nullptr;
    std::shared_ptr<LossTask> loss_task = nullptr;
    std::vector<std::shared_ptr<LayerTask>> layers = {};
};

#endif
