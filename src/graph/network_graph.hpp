#ifndef GRAPH_NETWORK_GRAPH_H
#define GRAPH_NETWORK_GRAPH_H
#include "../state/pipeline_state_manager.hpp"
#include "../state/optimizer_state_manager.hpp"
#include "../task/optimizer/optimizer_task.hpp"
#include "../task/fwd_task.hpp"
#include "../task/bwd_task.hpp"
#include "../task/loss/loss_task.hpp"
#include "../tools/timer.hpp"
#include <hedgehog/hedgehog.h>

#define NetworkGraphIn InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphOut InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphIO 2, NetworkGraphIn, NetworkGraphOut

class NetworkGraph : public hh::Graph<NetworkGraphIO> {
  public:
    NetworkGraph(size_t nb_shards = 1) : hh::Graph<NetworkGraphIO>(),
    nb_shards_(nb_shards) {
        auto pipeline = std::make_shared<PipelineState>();
        pipeline_state_ = std::make_shared<PipelineStateManager>(pipeline);
        optimizer_state_ = std::make_shared<OptimizerStateManager>(
            std::make_shared<OptimizerState>(), pipeline);

        this->inputs(pipeline_state_);
        this->outputs(pipeline_state_);

        // add the task for the shards
        for (size_t i = 0; i < nb_shards_; ++i) {
            fwds_.push_back(std::make_shared<FwdTask>());
            bwds_.push_back(std::make_shared<BwdTask>());
        }
    }

    void add_layer(std::shared_ptr<Layer<ftype>> layer, size_t shard_idx = 0) {
        layer->idx = layer_idx++;
        fwds_[shard_idx]->add_layer(layer);
        bwds_[shard_idx]->add_layer(layer);
        layers_.push_back(layer);
    }

    // TODO: the loss should be a functor
    void set_loss(std::shared_ptr<LossTask> loss_task) {
        this->loss_task_ = loss_task;
    }

    // TODO: the optimizer should be a functor
    void set_optimizer(std::shared_ptr<OptimizerTask> optimizer_task) {
        this->optimizer_task_ = optimizer_task;
    }

    void build() {
        // there are a lot of communcations issues
        //
        // we need a way to control sender and receivers more clearly. Perhaps
        // having a copy of each task is a good idea (one copy serving for the
        // fwd pass and the other for the bwd pass).
        //
        // It will be easire if the layers are not task but just functors. In
        // this case we should have generic FwdTask and BwdTask that will be
        // create here when a layer is added. These classes will hold the
        // functors and use theme for the fwd and bwd passes. Each functor
        // should have a Fwd and Bwd functions that return the appropriate data
        // to the task.
        //
        // The end user will not manage HH communications.
        //
        // If we do this, there might be some areas where we could optimize the
        // communications. For instance, there is no need for any communications
        // durint the fwd and bwd passes.
        //
        // Well this removes the possiblity of using HH for transfering data
        // between nodes inside the network. In this case we want shards to be
        // encapsulated in a task that will run on a given devise. This will not
        // be completely opaque to the user since it will be required to declare
        // the shards by hand (which is greate).
        //
        // Final thought:
        // Use functors for layers and create generic tasks that will call the
        // functors functions for the fwd and bwd passes.
        // This generic tasks will be responsible for managing a shared
        // (collection of layers).
        // The result is less classes so less communications. Moreover, since
        // there will be dedicated tasks for the fwd and bwd passes, we solve
        // the communcation conflict issue.


        // connect the fwds tasks
        this->edges(pipeline_state_, fwds_.front());
        for (size_t i = 0; i < fwds_.size() - 1; ++i) {
            this->edges(fwds_[i], fwds_[i + 1]);
        }
        this->edges(fwds_.back(), pipeline_state_);

        // connect loss and bwds tasks
        this->edges(pipeline_state_, loss_task_);
        this->edges(loss_task_, bwds_.back());
        for (size_t i = bwds_.size() - 1; i > 1; --i) {
            this->edges(bwds_[i], bwds_[i - 1]);
        }
        this->edges(bwds_.front(), pipeline_state_);

        // connect optimizer
        this->edges(pipeline_state_, optimizer_state_);
        this->edges(optimizer_state_, pipeline_state_);
        this->edges(optimizer_state_, optimizer_task_);
        this->edges(optimizer_task_, optimizer_state_);
    }

    void init_network_state(NetworkState<ftype> &state) {
        timer_start(graph_init);
        state.layer_states = std::vector<LayerState<ftype>>(layers_.size());
        for (auto &layer : layers_) {
            layer->init(state);
        }
        optimizer_task_->init(state);
        loss_task_->init(state);
        timer_end(graph_init);
        timer_report_prec(graph_init, milliseconds);
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
    std::vector<std::shared_ptr<FwdTask>> fwds_ = {};
    std::vector<std::shared_ptr<BwdTask>> bwds_ = {};
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
    size_t nb_shards_ = 0;
    size_t layer_idx = 0;
};

#endif
