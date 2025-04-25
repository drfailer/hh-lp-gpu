#ifndef TASK_BWD_TASK_H
#define TASK_BWD_TASK_H
#include "../data/bwd_data.hpp"
#include "../layers/layer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define BwdTaskIn BwdData<ftype>
#define BwdTaskOut BwdData<ftype>
#define BwdTaskType 1, BwdTaskIn, BwdTaskOut

class BwdTask : public hh::AbstractCUDATask<BwdTaskType> {
  public:
    BwdTask() : hh::AbstractCUDATask<BwdTaskType>("BwdTask", 1) {}

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        ftype *error = data->error;
        auto &states = data->states;

        for (int i = layers_.size() - 1; i >= 0; --i) {
            error = layers_[i]->bwd(states, error);
        }
        data->error = error;
        this->addResult(data);
    }

    void add_layer(std::shared_ptr<Layer<ftype>> layer) {
        layers_.push_back(layer);
    }

    std::shared_ptr<hh::AbstractTask<BwdTaskType>> copy() override {
        throw std::logic_error("error: BwdTask should not be copied.");
    }

  private:
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
};

#endif
