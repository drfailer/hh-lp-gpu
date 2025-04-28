#ifndef STATE_OPTIMIZER_STATE_H
#define STATE_OPTIMIZER_STATE_H
#include "../data/opt_data.hpp"
#include "../data/opt_layer_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerStateIn OptLayerData<ftype>
#define OptimizerStateOut OptData<ftype>
#define OptimizerStateIO 1, OptimizerStateIn, OptimizerStateOut

class OptimizerState : public hh::AbstractState<OptimizerStateIO> {
  public:
    OptimizerState() : hh::AbstractState<OptimizerStateIO>() {}

    void execute(std::shared_ptr<OptLayerData<ftype>> data) override {
        ++nb_processed_layers_;

        if (nb_processed_layers_ == nb_layers_) {
            // all the layers have been updated and optimized
            nb_processed_layers_ = 0;
            this->addResult(std::make_shared<OptData<ftype>>(
                data->state, data->learning_rate));
        }
    }

    void nb_layers(size_t n) { this->nb_layers_ = n; }

  private:
    size_t nb_processed_layers_ = 0;
    size_t nb_layers_ = 0;
};

#endif
