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

    void init(std::shared_ptr<NNState<ftype>> state) {
        auto model_output = state->layers.back().output;
        state->loss.tensor = create_tensor<ftype>(model_output->dims());
    }

    void execute(std::shared_ptr<LossFwdData<ftype>> data) override {
        loss_->fwd(data->states->loss, data->input, data->ground_truth);
        this->addResult(data);
    }

    void execute(std::shared_ptr<LossBwdData<ftype>> data) override {
        loss_->bwd(data->states->loss, data->input, data->ground_truth);
        this->addResult(std::make_shared<BwdData<ftype>>(
            data->states, data->states->loss.tensor, data->learning_rate));
    }

  private:
    std::shared_ptr<Loss<ftype>> loss_ = nullptr;
};

#endif
