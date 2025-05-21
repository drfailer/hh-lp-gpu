#ifndef TASK_LOSS_TASK_H
#define TASK_LOSS_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/loss_fwd_data.hpp"
#include "../model/loss/loss.hpp"
#include "../tools/gpu.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LossTaskIn LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskOut LossFwdData<ftype>, BwdData<ftype>
#define LossTaskIO 2, LossTaskIn, LossTaskOut

class LossTask : public hh::AbstractCUDATask<LossTaskIO> {
  public:
    LossTask(std::shared_ptr<Loss<ftype>> loss) : loss_(loss) {}

    void create_state(ftype **mem, int64_t output_size) {
        CUDA_CHECK(alloc_gpu(mem, (size_t)output_size));
    }

    void init(int64_t output_size) { loss_->init(output_size); }

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
