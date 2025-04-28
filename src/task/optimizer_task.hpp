#ifndef TASK_OPTIMIZER_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_OPTIMIZER_TASK_H
#include "../data/opt_layer_data.hpp"
#include "../optimizers/optimizer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerTaskIn OptLayerData<ftype>
#define OptimizerTaskOut OptLayerData<ftype>
#define OptimizerTaskIO 1, OptimizerTaskIn, OptimizerTaskOut

class OptimizerTask : public hh::AbstractCUDATask<OptimizerTaskIO> {
  public:
    using OptimizerList = std::vector<std::shared_ptr<Optimizer<ftype>>>;

  public:
    OptimizerTask(std::shared_ptr<Optimizer<ftype>> optimizer,
                  size_t nb_threads)
        : hh::AbstractCUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizer_(optimizer) {}

    OptimizerTask(std::shared_ptr<OptimizerList> optimizers, size_t nb_threads)
        : hh::AbstractCUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizers_(optimizers) {}

    void init(NetworkState<ftype> const &state) {
        optimizers_ = std::make_shared<OptimizerList>();
        for (size_t i = 0; i < state.layer_states.size(); ++i) {
            auto optimizer_layer_i = optimizer_->copy();
            optimizer_layer_i->init(state.layer_states[i]);
            optimizers_->push_back(std::move(optimizer_layer_i));
        }
    }

    void execute(std::shared_ptr<OptLayerData<ftype>> data) override {
        optimizers_->operator[](data->idx)->optimize(
            data->state.layer_states[data->idx], data->learning_rate);
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<OptimizerTaskIO>> copy() override {
        return std::make_shared<OptimizerTask>(optimizers_,
                                               this->numberThreads());
    }

  private:
    std::shared_ptr<Optimizer<ftype>> optimizer_ = nullptr;
    std::shared_ptr<OptimizerList> optimizers_ = nullptr;
};

#endif
