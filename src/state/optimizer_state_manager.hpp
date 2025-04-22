#ifndef STATE_OPTIMIZER_STATE_MANAGER_H
#define STATE_OPTIMIZER_STATE_MANAGER_H
#include "optimizer_state.hpp"
#include <hedgehog/hedgehog.h>

class OptimizerStateManager : public hh::StateManager<OptimizerStateIO> {
  public:
    OptimizerStateManager(std::shared_ptr<OptimizerState> const &state)
        : hh::StateManager<OptimizerStateIO>(state, "OptimizerState") {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret =
            std::dynamic_pointer_cast<OptimizerState>(this->state())->isDone();
        this->state()->unlock();
        return ret;
    }
};

#endif
