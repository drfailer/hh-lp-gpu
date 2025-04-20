#ifndef STATE_INFERENCE_STATE_MANAGER_H
#define STATE_INFERENCE_STATE_MANAGER_H
#include "inference_state.hpp"
#include <hedgehog/hedgehog.h>

class InferenceStateManager : public hh::StateManager<InferenceStateIO> {
  public:
    InferenceStateManager(std::shared_ptr<InferenceState> const &state)
        : hh::StateManager<InferenceStateIO>(state, "InferenceState") {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret =
            std::dynamic_pointer_cast<InferenceState>(this->state())->isDone();
        this->state()->unlock();
        return ret;
    }
};

#endif
