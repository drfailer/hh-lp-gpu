#ifndef STATE_TRAINING_STATE_H
#define STATE_TRAINING_STATE_H
#include "../data/bwd_data.hpp"
#include "../data/fwd_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/opt_data.hpp"
#include "../data/training_data.hpp"
#include "../data/update_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define from_step_to(curr, to)                                                 \
    if (state.step != curr) {                                                  \
        std::cerr << "error: entering train step " #to " from step " #curr "." \
                  << std::endl;                                                \
        return;                                                                \
    }                                                                          \
    state.step = to;

#define TrainingStateIn                                                        \
    TrainingData<ftype>, FwdData<ftype>, LossBwdData<ftype>, BwdData<ftype>,   \
        OptData<ftype>, UpdateData<ftype>
#define TrainingStateOut                                                       \
    TrainingData<ftype>, FwdData<ftype>, LossBwdData<ftype>, BwdData<ftype>,   \
        OptData<ftype>, UpdateData<ftype>
#define TrainingStateIO 6, TrainingStateIn, TrainingStateOut

class TrainingState : public hh::AbstractState<TrainingStateIO> {
  public:
    TrainingState() : hh::AbstractState<TrainingStateIO>() {}

  public:
    enum class Steps {
        Idle,
        Fwd,
        LossBwd,
        Bwd,
        Opt,
        Update,
    };

  public:
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
                train_data.data_set.datas[state.data_set_idx++].input));
        }
    }

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        from_step_to(Steps::Fwd, Steps::LossBwd);

        this->addResult(std::make_shared<LossBwdData<ftype>>(
            data->states, data->input,
            train_data.data_set.datas[state.data_set_idx].ground_truth,
            nullptr));
    }

    void execute(std::shared_ptr<LossBwdData<ftype>> data) override {
        from_step_to(Steps::LossBwd, Steps::Bwd);

        this->addResult(
            std::make_shared<BwdData<ftype>>(data->states, data->error));
    }

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        from_step_to(Steps::Bwd, Steps::Opt);

        this->addResult(std::make_shared<OptData<ftype>>(data->states));
    }

    void execute(std::shared_ptr<OptData<ftype>> data) override {
        from_step_to(Steps::Opt, Steps::Update);

        this->addResult(std::make_shared<UpdateData<ftype>>(
            data->states, train_data.learning_rate));
    }

    // TODO: the update should be done in separated state so all the update can
    // be done in parallel
    void execute(std::shared_ptr<UpdateData<ftype>> data) override {
        from_step_to(Steps::Update, Steps::Fwd);

        if (state.epoch < train_data.epochs &&
            state.data_set_idx < train_data.data_set.datas.size()) {
            this->addResult(std::make_shared<FwdData<ftype>>(
                data->states,
                train_data.data_set.datas[state.data_set_idx++].input));
        } else {
            this->addResult(std::make_shared<TrainingData<ftype>>(
                data->states, train_data.data_set, train_data.learning_rate,
                train_data.epochs));
        }
    }

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
