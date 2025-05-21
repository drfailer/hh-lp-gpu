#ifndef TASK_OPTIMIZER_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_OPTIMIZER_TASK_H
#include "../data/opt_layer_data.hpp"
#include "../model/optimizer/optimizer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerTaskIn OptLayerData<ftype>
#define OptimizerTaskOut OptLayerData<ftype>
#define OptimizerTaskIO 1, OptimizerTaskIn, OptimizerTaskOut

class OptimizerTask : public hh::AbstractCUDATask<OptimizerTaskIO> {
  public:
    using OptimizerList = std::vector<std::shared_ptr<Optimizer<ftype>>>;

  public:
    OptimizerTask(size_t nb_threads)
        : hh::AbstractCUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizers_(std::make_shared<OptimizerList>()) {}

    OptimizerTask(std::shared_ptr<OptimizerList> optimizers, size_t nb_threads)
        : hh::AbstractCUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizers_(optimizers) {}

    void execute(std::shared_ptr<OptLayerData<ftype>> data) override {
        optimizers_->operator[](data->idx)->optimize(
            data->state.layers[data->idx], data->learning_rate);
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<OptimizerTaskIO>> copy() override {
        return std::make_shared<OptimizerTask>(optimizers_,
                                               this->numberThreads());
    }

    void add_layer(std::shared_ptr<Optimizer<ftype>> layer_optimizer) {
        optimizers_->push_back(layer_optimizer);
    }

  private:
    std::shared_ptr<OptimizerList> optimizers_ = nullptr;
};

#endif
