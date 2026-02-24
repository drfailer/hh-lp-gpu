#ifndef STATE_PIPELINE_STATE_H
#define STATE_PIPELINE_STATE_H
#include "../data/fwd_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/opt_data.hpp"
#include "../data/prediction_data.hpp"
#include "../data/training_data.hpp"
#include "../types.hpp"
#include "state.hpp"
#include <hedgehog/hedgehog.h>
#include "../tools/log.h"

#define PipelineStateIn                                                        \
    TrainingData, PredictionData, FwdData, OptData
#define PipelineStateOut                                                       \
    TrainingData, PredictionData, FwdData,                                     \
        LossBwdData, OptData
#define PipelineStateIO 4, PipelineStateIn, PipelineStateOut

class PipelineState : public hh::AbstractState<PipelineStateIO> {
  public:
    PipelineState() : hh::AbstractState<PipelineStateIO>() {}

  public:
    enum class Step_ {
        Idle,
        Inference,
        Fwd,
        Bwd,
        Opt,
        Finish,
    };

  public:
    void execute(std::shared_ptr<PredictionData> data) override {
        step_from_to(Step_::Idle, Step_::Inference);
        this->addResult(
            std::make_shared<FwdData>(data->states, data->input));
    }

    void execute(std::shared_ptr<TrainingData> data) override {
        step_from_to(Step_::Idle, Step_::Fwd);
        // init
        train_data_.data_set = &data->data_set;
        train_data_.epochs = data->epochs;

        // start computation
        if (data_set_idx_ < train_data_.data_set->datas.size()) {
            this->addResult(std::make_shared<FwdData>(
                data->states,
                &train_data_.data_set->datas[data_set_idx_].input));
        }
    }

    void execute(std::shared_ptr<FwdData> data) override {
        if (step_ == Step_::Fwd) {
            // we might remove this
            step_from_to(Step_::Fwd, Step_::Bwd);
            this->addResult(std::make_shared<LossBwdData>(
                data->network_data, data->input,
                &train_data_.data_set->datas[data_set_idx_].ground_truth));
        } else {
            step_from_to(Step_::Inference, Step_::Idle);
            this->addResult(std::make_shared<PredictionData>(
                data->network_data, data->input));
        }
    }

    void execute(std::shared_ptr<OptData> data) override {
        ++data_set_idx_;
        // TODO: add a log rate and compute the loss
        // if (state.data_set_idx % 1'000 == 0) std::cout << state.data_set_idx
        // << std::endl;
        if (data_set_idx_ >= train_data_.data_set->datas.size()) {
            // if (state.data_set_idx >= 2) {
            printf("epoch %ld\n", epoch_);
            INFO_GRP("new epoch", INFO_GRP_PIPELINE_STEP);
            data_set_idx_ = 0;
            ++epoch_;
        }

        if (epoch_ < train_data_.epochs) {
            step_from_to(Step_::Bwd, Step_::Fwd);
            this->addResult(std::make_shared<FwdData>(
                data->states,
                &train_data_.data_set->datas[data_set_idx_].input));
        } else {
            step_from_to(Step_::Bwd, Step_::Idle);
            this->addResult(std::make_shared<TrainingData>(
                data->states, *train_data_.data_set, train_data_.epochs));
        }
    }

  public:
    bool done() const { return step_ == Step_::Finish; }

    void clean() override {
        step_ = Step_::Idle;
        epoch_ = 0;
        data_set_idx_ = 0;
        train_data_ = {0};
    }

    void terminate() { step_ = Step_::Finish; }

  private:
    size_t epoch_ = 0;
    size_t data_set_idx_ = 0;
    Step_ step_ = Step_::Idle;
    struct {
        size_t epochs = 0;
        DataSet *data_set;
    } train_data_;
};

#endif
