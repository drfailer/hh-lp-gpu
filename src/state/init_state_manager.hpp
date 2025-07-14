#ifndef STATE_INIT_STATE_MANAGER
#define STATE_INIT_STATE_MANAGER
#include "init_state.hpp"
#include <hedgehog/hedgehog.h>

class InitStateManager : public hh::StateManager<InitStateIO> {
  public:
    InitStateManager(std::shared_ptr<InitState> const &state)
        : hh::StateManager<InitStateIO>(state, "InitState") {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret = std::dynamic_pointer_cast<InitState>(this->state())->done();
        this->state()->unlock();
        return ret;
    }
};

#endif
