#ifndef STATE_OPTIMIZER_STATE_MANAGER_H
#define STATE_OPTIMIZER_STATE_MANAGER_H
#include "optimizer_state.hpp"
#include "pipeline_state.hpp"
#include <hedgehog/hedgehog.h>

class OptimizerStateManager : public hh::StateManager<OptimizerStateIO> {
  public:
    OptimizerStateManager(std::shared_ptr<OptimizerState> const &state,
            std::shared_ptr<PipelineState> pipeline_state)
        : hh::StateManager<OptimizerStateIO>(state, "OptimizerState"),
        pipeline_state_(pipeline_state) {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret = pipeline_state_->isDone();
        this->state()->unlock();
        return ret;
    }

private:
    std::shared_ptr<PipelineState> pipeline_state_ = nullptr;
};

#endif
