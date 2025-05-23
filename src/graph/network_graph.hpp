#ifndef GRAPH_NETWORK_GRAPH_H
#define GRAPH_NETWORK_GRAPH_H
#include "../state/optimizer_state_manager.hpp"
#include "../state/pipeline_state_manager.hpp"
#include "../task/bwd_task.hpp"
#include "../task/fwd_task.hpp"
#include "../task/loss_task.hpp"
#include "../task/optimizer_task.hpp"
#include "../tools/timer.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define NetworkGraphIn InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphOut InferenceData<ftype>, TrainingData<ftype>
#define NetworkGraphIO 2, NetworkGraphIn, NetworkGraphOut

class NetworkGraph : public hh::Graph<NetworkGraphIO> {
  public:
    NetworkGraph()
        : hh::Graph<NetworkGraphIO>(),
          pipeline_(std::make_shared<PipelineState>()),
          pipeline_state_(std::make_shared<PipelineStateManager>(pipeline_)),
          optimizer_(std::make_shared<OptimizerState>()),
          optimizer_state_(
              std::make_shared<OptimizerStateManager>(optimizer_, pipeline_)) {
        // TODO: we might want a minibatch generator as input
        this->inputs(pipeline_state_);
        this->outputs(pipeline_state_);

        fwds_.push_back(std::make_shared<FwdTask>());
        bwds_.push_back(std::make_shared<BwdTask>());
    }

  public:
    void add_layer(std::shared_ptr<Layer<ftype>> layer) {
        layer->idx = layer_idx_++;
        fwds_.back()->add_layer(layer);
        bwds_.back()->add_layer(layer);
        layers_.push_back(layer);
    }

    template <typename LayerType, typename... Types>
    void add_layer(Types... args) {
        add_layer(std::make_shared<LayerType>(std::forward<Types>(args)...));
    }

    void cut_layer() {
        if (fwds_.back()->layers().size() == 0) {
            throw std::logic_error(
                "error: cannot add a cut layer after in an empty shard.");
        }
        fwds_.push_back(std::make_shared<FwdTask>());
        bwds_.push_back(std::make_shared<BwdTask>());
    }

    template <typename LossType, typename... Types>
    void set_loss(Types... args) {
        this->loss_task_ = std::make_shared<LossTask>(
            std::make_shared<LossType>(std::forward<Types>(args)...));
    }

    template <typename OptimizerType, typename... Types>
    void set_optimizer(size_t nb_threads, Types... args) {
        this->optimizer_task_ = std::make_shared<OptimizerTask>(nb_threads);
        this->optimizer_factory_ =
            std::make_shared<OptimizerType>(std::forward<Types>(args)...);
    }

    void build() {
        optimizer_->nb_layers(layers_.size());

        // connect the fwds tasks
        this->edges(pipeline_state_, fwds_.front());
        for (size_t i = 0; i < fwds_.size() - 1; ++i) {
            this->edges(fwds_[i], fwds_[i + 1]);
        }
        this->edges(fwds_.back(), pipeline_state_);

        // connect loss and bwds tasks
        if (loss_task_) {
            this->edges(pipeline_state_, loss_task_);
            this->edges(loss_task_, bwds_.back());
            for (size_t i = bwds_.size() - 1; i >= 1; --i) {
                this->edges(bwds_[i], bwds_[i - 1]);
            }
        }

        // connect optimizer
        if (optimizer_task_) {
            for (size_t i = 0; i < bwds_.size(); ++i) {
                this->edges(bwds_[i], optimizer_task_);
            }
            this->edges(optimizer_task_, optimizer_state_);
            this->edges(optimizer_state_, pipeline_state_);
        }
    }

    void init(NetworkState<ftype> &state, int64_t batch_count) {
        for (auto layer : layers_) {
            layer->init(state, batch_count);
            optimizer_task_->add_layer(optimizer_factory_->create());
        }
         loss_task_->init(state);
    }

    void terminate() {
        pipeline_->terminate();
        this->finishPushingData();
        this->waitForTermination();
    }

  public:
    NetworkState<ftype> create_state() {
        NetworkState<ftype> state;

        timer_start(create_state);
        state.layers = std::vector<layer_state_t<ftype>>(layers_.size());
        for (auto &layer : layers_) {
            layer->create_state(state);
        }
        timer_end(create_state);
        timer_report_prec(create_state, milliseconds);
        return state;
    }

    void destroy_state(NetworkState<ftype> &state) {
        for (auto &layer_state : state.layers) {
            destroy_layer_state(layer_state);
        }
        delete state.loss;
    }

  public:
    template <typename OutType> std::shared_ptr<OutType> get() {
        return std::get<std::shared_ptr<OutType>>(*this->getBlockingResult());
    }

  private:
    std::shared_ptr<PipelineState> pipeline_ = nullptr;
    std::shared_ptr<PipelineStateManager> pipeline_state_ = nullptr;
    std::shared_ptr<LossTask> loss_task_ = nullptr;
    std::shared_ptr<Loss<ftype>> loss_ = nullptr;
    std::shared_ptr<OptimizerTask> optimizer_task_ = nullptr;
    std::shared_ptr<OptimizerState> optimizer_ = nullptr;
    std::shared_ptr<OptimizerStateManager> optimizer_state_ = nullptr;
    std::shared_ptr<Optimizer<ftype>> optimizer_factory_ = nullptr;
    std::vector<std::shared_ptr<FwdTask>> fwds_ = {};
    std::vector<std::shared_ptr<BwdTask>> bwds_ = {};
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
    size_t layer_idx_ = 0;
};

#endif
