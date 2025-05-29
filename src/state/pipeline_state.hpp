#ifndef STATE_PIPELINE_STATE_H
#define STATE_PIPELINE_STATE_H
#include "../data/fwd_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/opt_data.hpp"
#include "../data/prediction_data.hpp"
#include "../data/training_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>
#include <log.h/log.h>

#define from_step_to(curr, to)                                                 \
    INFO_GRP("from stpe " #curr " to " #to ".", INFO_GRP_PIPELINE_STEP);       \
    if (state.step != curr) {                                                  \
        std::cerr << "error: entering train step " #to " from step " #curr "." \
                  << std::endl;                                                \
        return;                                                                \
    }                                                                          \
    state.step = to;

#define PipelineStateIn                                                        \
    TrainingData<ftype>, PredictionData<ftype>, FwdData<ftype>, OptData<ftype>
#define PipelineStateOut                                                       \
    TrainingData<ftype>, PredictionData<ftype>, FwdData<ftype>,                \
        LossBwdData<ftype>, OptData<ftype>
#define PipelineStateIO 4, PipelineStateIn, PipelineStateOut

class PipelineState : public hh::AbstractState<PipelineStateIO> {
  public:
    PipelineState() : hh::AbstractState<PipelineStateIO>() {}

  public:
    enum class Steps {
        Idle,
        Inference,
        Fwd,
        Bwd,
        Opt,
        Finish,
    };

  public:
    void execute(std::shared_ptr<PredictionData<ftype>> data) override {
        from_step_to(Steps::Idle, Steps::Inference);
        this->addResult(
            std::make_shared<FwdData<ftype>>(data->states, data->input));
    }

    void execute(std::shared_ptr<TrainingData<ftype>> data) override {
        from_step_to(Steps::Idle, Steps::Fwd);
        // init
        train_data.data_set = data->data_set;
        train_data.epochs = data->epochs;
        train_data.learning_rate = data->learning_rate;

        // start computation
        if (state.data_set_idx < train_data.data_set.datas.size()) {
            this->addResult(std::make_shared<FwdData<ftype>>(
                data->states,
                train_data.data_set.datas[state.data_set_idx].input));
        }
    }

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        if (state.step == Steps::Fwd) {
            // we might remove this
            from_step_to(Steps::Fwd, Steps::Bwd);
            this->addResult(std::make_shared<LossBwdData<ftype>>(
                data->states, data->input,
                train_data.data_set.datas[state.data_set_idx].ground_truth,
                nullptr, train_data.learning_rate));
        } else {
            from_step_to(Steps::Inference, Steps::Idle);
            this->addResult(std::make_shared<PredictionData<ftype>>(
                data->states, data->input));
        }
    }

    void execute(std::shared_ptr<OptData<ftype>> data) override {
        ++state.data_set_idx;
        // TODO: add a log rate and compute the loss
        // if (state.data_set_idx % 1'000 == 0) std::cout << state.data_set_idx
        // << std::endl;
        if (state.data_set_idx >= train_data.data_set.datas.size()) {
            INFO_GRP("new epoch", INFO_GRP_PIPELINE_STEP);
            state.data_set_idx = 0;
            ++state.epoch;
        }

        if (state.epoch < train_data.epochs) {
            from_step_to(Steps::Bwd, Steps::Fwd);
            this->addResult(std::make_shared<FwdData<ftype>>(
                data->states,
                train_data.data_set.datas[state.data_set_idx].input));
        } else {
            from_step_to(Steps::Bwd, Steps::Idle);
            this->addResult(std::make_shared<TrainingData<ftype>>(
                data->states, train_data.data_set, train_data.learning_rate,
                train_data.epochs));
        }
    }

  public:
    bool isDone() const { return state.step == Steps::Finish; }

    void clean() override {
        state.step = Steps::Idle;
        state.epoch = 0;
        state.data_set_idx = 0;
        train_data = {0};
    }

    void terminate() { state.step = Steps::Finish; }

  private:
    struct {
        Steps step = Steps::Idle;
        size_t epoch = 0;
        size_t data_set_idx = 0;
    } state;
    struct {
        size_t epochs = 0;
        ftype learning_rate = 0;
        DataSet<ftype> data_set;
    } train_data;
};

#endif
