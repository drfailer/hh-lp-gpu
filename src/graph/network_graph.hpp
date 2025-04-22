#ifndef GRAPH_NETWORK_GRAPH_H
#define GRAPH_NETWORK_GRAPH_H
#include "../state/pipeline_state_manager.hpp"
#include "../state/optimizer_state_manager.hpp"
#include "../task/optimizer/optimizer_task.hpp"
#include "../task/layer_task.hpp"
#include "../task/loss/loss_task.hpp"
#include <hedgehog/hedgehog.h>

#define NetworkGraphIn InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphOut InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphIO 2, NetworkGraphIn, NetworkGraphOut

class NetworkGraph : public hh::Graph<NetworkGraphIO> {
  public:
    NetworkGraph() : hh::Graph<NetworkGraphIO>() {
        pipeline_state_ = std::make_shared<PipelineStateManager>(
            std::make_shared<PipelineState>());
        optimizer_state_ = std::make_shared<OptimizerStateManager>(
            std::make_shared<OptimizerState>());

        this->inputs(pipeline_state_);
        this->outputs(pipeline_state_);
    }

    void add_layer(std::shared_ptr<LayerTask> layer) {
        layers_.emplace_back(layer);
    }

    void set_loss(std::shared_ptr<LossTask> loss_task) {
        this->loss_task_ = loss_task;
    }

    void set_optimizer(std::shared_ptr<OptimizerTask> optimizer_task) {
        this->optimizer_task_ = optimizer_task;
    }

    void build() {
        this->edges(pipeline_state_, layers_.front());
        for (size_t i = 0; i < layers_.size() - 1; ++i) {
            this->edges(layers_[i], layers_[i + 1]);
        }
        this->edges(layers_.back(), pipeline_state_);
        // pipeline_state <-> loss_task
        this->edges(pipeline_state_, loss_task_);
        this->edges(loss_task_, pipeline_state_);
        // pipeline_state <-> optimizer_state
        this->edges(pipeline_state_, optimizer_state_);
        this->edges(optimizer_state_, pipeline_state_);
        // optimizer_state <-> optimizer_task
        this->edges(optimizer_state_, optimizer_task_);
        this->edges(optimizer_task_, optimizer_state_);
    }

    void init_network_state(NetworkState<ftype> &state) {
        state.layer_states = std::vector<LayerState<ftype>>(layers_.size());
        for (auto layer : layers_) {
            layer->init(state);
        }
        loss_task_->init(state);
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
    std::shared_ptr<PipelineStateManager> pipeline_state_ = nullptr;
    std::shared_ptr<LossTask> loss_task_ = nullptr;
    std::shared_ptr<OptimizerTask> optimizer_task_ = nullptr;
    std::shared_ptr<OptimizerStateManager> optimizer_state_ = nullptr;
    std::vector<std::shared_ptr<LayerTask>> layers_ = {};
};

#endif
