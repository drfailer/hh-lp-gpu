#ifndef STATE_INIT_STATE
#define STATE_INIT_STATE
#include "../data/init_data.hpp"
#include "../data/init_parameters_data.hpp"
#include "../types.hpp"
#include "state.hpp"
#include <hedgehog/hedgehog.h>

#define InitStateIn                                                            \
    InitParametersData<ftype, InitTarget::Network>,                            \
        InitData<ftype, InitTarget::Network>,                                  \
        InitParametersData<ftype, InitTarget::Layer>,                          \
        InitData<ftype, InitTarget::Layer>, InitData<ftype, InitTarget::Loss>
#define InitStateOut                                                           \
    InitParametersData<ftype, InitTarget::Network>,                            \
        InitData<ftype, InitTarget::Network>,                                  \
        InitParametersData<ftype, InitTarget::Layer>,                          \
        InitData<ftype, InitTarget::Layer>, InitData<ftype, InitTarget::Loss>
#define InitStateIO 5, InitStateIn, InitStateOut

class InitState : public hh::AbstractState<InitStateIO> {
  public:
    InitState() : hh::AbstractState<InitStateIO>() {}

  public:
    enum class Step {
        Idle,
        CreateParameters,
        InitLayer,
        InitLoss,
        Finish,
    };

  public:
    void execute(std::shared_ptr<InitParametersData<ftype, InitTarget::Network>>
                     data) override {
        step_from_to(Step::Idle, Step::CreateParameters);
        this->addResult(
            std::make_shared<InitParametersData<ftype, InitTarget::Layer>>(
                data->states));
    }

    void execute(std::shared_ptr<InitParametersData<ftype, InitTarget::Layer>>
                     data) override {
        step_from_to(Step::CreateParameters, Step::Idle);
        this->addResult(
            std::make_shared<InitParametersData<ftype, InitTarget::Network>>(
                data->states));
    }

    void execute(
        std::shared_ptr<InitData<ftype, InitTarget::Network>> data) override {
        step_from_to(Step::Idle, Step::InitLayer);
        this->addResult(std::make_shared<InitData<ftype, InitTarget::Layer>>(
            data->network_data, data->input_dims));
    }

    void
    execute(std::shared_ptr<InitData<ftype, InitTarget::Layer>> data) override {
        if (!has_loss) {
            step_from_to(Step::InitLayer, Step::Idle);
            this->addResult(
                std::make_shared<InitData<ftype, InitTarget::Network>>(
                    data->network_data, data->input_dims));
        } else {
            step_from_to(Step::InitLayer, Step::InitLoss);
            this->addResult(std::make_shared<InitData<ftype, InitTarget::Loss>>(
                data->network_data, data->input_dims));
        }
    }

    void
    execute(std::shared_ptr<InitData<ftype, InitTarget::Loss>> data) override {
        step_from_to(Step::InitLoss, Step::Idle);
        this->addResult(std::make_shared<InitData<ftype, InitTarget::Network>>(
            data->network_data, data->input_dims));
    }

    void terminate() { step_ = Step::Finish; }

    bool done() const { return step_ == Step::Finish; }

    void clean() override { step_ = Step::Idle; }

  public:
    bool has_loss = false;

  private:
    Step step_ = Step::Idle;
};

#endif
