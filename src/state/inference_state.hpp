#ifndef STATE_INFERENCE_STATE_H
#define STATE_INFERENCE_STATE_H
#include <hedgehog/hedgehog.h>
#include "../types.hpp"
#include "../data/fwd_data.hpp"
#include "../data/inference_data.hpp"

#define InferenceStateIn InferenceData<ftype>, FwdData<ftype>
#define InferenceStateOut InferenceData<ftype>, FwdData<ftype>
#define InferenceStateIO 2, InferenceStateIn, InferenceStateOut

class InferenceState : hh::AbstractState<InferenceStateIO> {
  public:
    InferenceState() : hh::AbstractState<InferenceStateIO>() {}

    void execute(std::shared_ptr<InferenceData<ftype>> data) override {
        this->addResult(
            std::make_shared<FwdData<ftype>>(data->states, data->input));
    }

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        this->addResult(
            std::make_shared<InferenceData<ftype>>(data->states, data->input));
    }
};

#endif
