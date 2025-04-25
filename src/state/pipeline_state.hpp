#ifndef STATE_PIPELINE_STATE_H
#define STATE_PIPELINE_STATE_H
#include "../data/bwd_data.hpp"
#include "../data/fwd_data.hpp"
#include "../data/inference_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/opt_data.hpp"
#include "../data/terminiate_data.hpp"
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
    TrainingData<ftype>, InferenceData<ftype>, FwdData<ftype>, BwdData<ftype>, \
        OptData<ftype>
#define PipelineStateOut                                                       \
    TrainingData<ftype>, InferenceData<ftype>, FwdData<ftype>,                 \
        LossBwdData<ftype>, BwdData<ftype>, OptData<ftype>, TerminateData
#define PipelineStateIO 5, PipelineStateIn, PipelineStateOut

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
    void execute(std::shared_ptr<InferenceData<ftype>> data) override {
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
            from_step_to(Steps::Fwd, Steps::Bwd);
            this->addResult(std::make_shared<LossBwdData<ftype>>(
                data->states, data->input,
                train_data.data_set.datas[state.data_set_idx].ground_truth,
                nullptr));
        } else {
            from_step_to(Steps::Inference, Steps::Finish);
            this->addResult(std::make_shared<InferenceData<ftype>>(
                data->states, data->input));
            this->addResult(std::make_shared<TerminateData>());
        }
    }

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        // TODO: make this appear somewhere v
        from_step_to(Steps::Bwd, Steps::Opt);

        this->addResult(std::make_shared<OptData<ftype>>(
            data->states, train_data.learning_rate));
    }

    void execute(std::shared_ptr<OptData<ftype>> data) override {
        ++state.data_set_idx;
        if (state.data_set_idx == train_data.data_set.datas.size()) {
            state.data_set_idx = 0;
            ++state.epoch;
            INFO("new epoch");
        }

        if (state.epoch < train_data.epochs) {
            from_step_to(Steps::Opt, Steps::Fwd);
            this->addResult(std::make_shared<FwdData<ftype>>(
                data->states,
                train_data.data_set.datas[state.data_set_idx].input));
        } else {
            from_step_to(Steps::Opt, Steps::Finish);
            this->addResult(std::make_shared<TrainingData<ftype>>(
                data->states, train_data.data_set, train_data.learning_rate,
                train_data.epochs));
            this->addResult(std::make_shared<TerminateData>());
        }
    }

  public:
    bool isDone() const { return state.step == Steps::Finish; }

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
