#ifndef STATE_PIPELINE_STATE_MANAGER_H
#define STATE_PIPELINE_STATE_MANAGER_H
#include "pipeline_state.hpp"
#include <hedgehog/hedgehog.h>

class PipelineStateManager : public hh::StateManager<PipelineStateIO> {
  public:
    PipelineStateManager(std::shared_ptr<PipelineState> const &state)
        : hh::StateManager<PipelineStateIO>(state, "PipelineState") {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret =
            std::dynamic_pointer_cast<PipelineState>(this->state())->isDone();
        this->state()->unlock();
        return ret;
    }
};

#endif
