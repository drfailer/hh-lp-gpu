#ifndef STATE_OPTIMIZER_STATE_H
#define STATE_OPTIMIZER_STATE_H
#include "../data/opt_data.hpp"
#include "../data/opt_layer_data.hpp"
#include "../data/terminiate_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerStateIn                                                       \
    OptData<ftype>, OptLayerData<ftype>, TerminateData
#define OptimizerStateOut OptData<ftype>, OptLayerData<ftype>
#define OptimizerStateIO 3, OptimizerStateIn, OptimizerStateOut

class OptimizerState : public hh::AbstractState<OptimizerStateIO> {
  public:
    OptimizerState() : hh::AbstractState<OptimizerStateIO>() {}

    void execute(std::shared_ptr<OptData<ftype>> data) override {
        size_t nb_layers = data->states.layer_states.size();

        isDone_ = false;
        data_ = data;
        nb_processed_layers_ = 0;
        for (size_t i = 0; i < nb_layers; ++i) {
            this->addResult(std::make_shared<OptLayerData<ftype>>(
                data->states.layer_states[i], data->learning_rate, i));
        }
    }

    void execute(std::shared_ptr<OptLayerData<ftype>>) override {
        assert(data_ != nullptr);
        ++nb_processed_layers_;

        if (nb_processed_layers_ == data_->states.layer_states.size()) {
            // all the layers have been updated and optimized
            this->addResult(data_);
        }
    }

    // receive this from the pipeline_state to indicate that the training is
    // over and the graph can terminiate.
    // TODO: this is not very clean.
    void execute(std::shared_ptr<TerminateData>) override {
        isDone_ = true;
    }

    bool isDone() const { return isDone_; }

  private:
    size_t nb_processed_layers_ = 0;
    std::shared_ptr<OptData<ftype>> data_ = nullptr;
    bool isDone_ = false;
};

#endif
