#ifndef TASK_OPTIMIZER_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_OPTIMIZER_TASK_H
#include "../data/opt_layer_data.hpp"
#include "../model/optimizer/optimizer.hpp"
#include "../types.hpp"
#include "cuda_task.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerTaskIn OptLayerData
#define OptimizerTaskOut OptLayerData
#define OptimizerTaskIO 1, OptimizerTaskIn, OptimizerTaskOut

class OptimizerTask : public CUDATask<OptimizerTaskIO> {
  public:
    using OptimizerList = std::vector<std::shared_ptr<Optimizer>>;

  public:
    OptimizerTask(std::shared_ptr<Optimizer> optimizer,
                  size_t nb_threads)
        : CUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizer_(optimizer) {}

    void execute(std::shared_ptr<OptLayerData> data) override {
        auto &ld = data->state->layers_datas[data->idx];
        optimizer_->optimize(cuda_, {ld.dw, ld.db}, {ld.w, ld.b});
        CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<OptimizerTaskIO>> copy() override {
        return std::make_shared<OptimizerTask>(optimizer_->copy(),
                                               this->numberThreads());
    }

  private:
    std::shared_ptr<Optimizer> optimizer_ = nullptr;
};

#endif
