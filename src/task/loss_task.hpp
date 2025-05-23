#ifndef TASK_LOSS_TASK_H
#define TASK_LOSS_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/loss_fwd_data.hpp"
#include "../model/loss/loss.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LossTaskIn LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskOut LossFwdData<ftype>, BwdData<ftype>
#define LossTaskIO 2, LossTaskIn, LossTaskOut

class LossTask : public hh::AbstractCUDATask<LossTaskIO> {
  public:
    LossTask(std::shared_ptr<Loss<ftype>> loss) : loss_(loss) {}

    void init(NetworkState<ftype> &state) {
        auto model_output = state.layers.back().output;
        state.loss = new Tensor<ftype>(model_output->dims(), model_output->strides());
    }

    void execute(std::shared_ptr<LossFwdData<ftype>> data) override {
        loss_->fwd(data->input, data->ground_truth, data->states.loss);
        this->addResult(data);
    }

    void execute(std::shared_ptr<LossBwdData<ftype>> data) override {
        loss_->bwd(data->input, data->ground_truth, data->states.loss);
        this->addResult(std::make_shared<BwdData<ftype>>(
            data->states, data->states.loss, data->learning_rate));
    }

  private:
    std::shared_ptr<Loss<ftype>> loss_ = nullptr;
};

#endif
