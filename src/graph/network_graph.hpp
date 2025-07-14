#ifndef GRAPH_NETWORK_GRAPH_H
#define GRAPH_NETWORK_GRAPH_H
#include "../state/init_state.hpp"
#include "../state/init_state_manager.hpp"
#include "../state/optimizer_state_manager.hpp"
#include "../state/pipeline_state_manager.hpp"
#include "../task/layer_tasks.hpp"
#include "../task/loss_task.hpp"
#include "../task/optimizer_task.hpp"
#include "../tools/timer.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define NetworkGraphIn                                                         \
    CreateParameterData<ftype>, InitData<ftype>, PredictionData<ftype>,        \
        TrainingData<ftype>
#define NetworkGraphOut                                                        \
    CreateParameterData<ftype>, InitData<ftype>, PredictionData<ftype>,        \
        TrainingData<ftype>
#define NetworkGraphIO 4, NetworkGraphIn, NetworkGraphOut

// TODO: the model should be separated from the graph
// TODO: add the optimizer in the BwdTask and remove OptimizerTask
class NetworkGraph : public hh::Graph<NetworkGraphIO> {
  public:
    NetworkGraph()
        : hh::Graph<NetworkGraphIO>(),
          init_state_(std::make_shared<InitState>()),
          init_state_manager_(std::make_shared<InitStateManager>(init_state_)),
          pipeline_state_(std::make_shared<PipelineState>()),
          pipeline_state_manager_(
              std::make_shared<PipelineStateManager>(pipeline_state_)),
          optimizer_state_(std::make_shared<OptimizerState>()),
          optimizer_state_manager_(std::make_shared<OptimizerStateManager>(
              optimizer_state_, pipeline_state_)) {
        this->inputs(pipeline_state_manager_);
        this->inputs(init_state_manager_);
        this->outputs(pipeline_state_manager_);
        this->outputs(init_state_manager_);

        CUDNN_CHECK(cudnnCreate(&cuda_data_.cudnn_handle));
    }

    ~NetworkGraph() { CUDNN_CHECK(cudnnDestroy(cuda_data_.cudnn_handle)); }

  public:
    template <typename LayerType, typename... Types>
    void add_layer(Types... args) {
        layer_tasks_.add_layer(
            std::make_shared<LayerType>(std::forward<Types>(args)...));
    }

    void cut_layer() { layer_tasks_.cut_layer(); }

    template <typename LossType, typename... Types>
    void set_loss(Types... args) {
        this->loss_task_ = std::make_shared<LossTask>(
            std::make_shared<LossType>(std::forward<Types>(args)...));
    }

    template <typename OptimizerType, typename... Types>
    void set_optimizer(size_t nb_threads, Types... args) {
        this->optimizer_task_ = std::make_shared<OptimizerTask>(
            std::make_shared<OptimizerType>(std::forward<Types>(args)...),
            nb_threads);
    }

  public:
    void build() {
        // connect the init tasks
        this->edges(init_state_manager_, layer_tasks_.inits.front());
        for (size_t i = 0; i < layer_tasks_.inits.size() - 1; ++i) {
            this->edges(layer_tasks_.inits[i], layer_tasks_.inits[i + 1]);
        }
        this->edges(layer_tasks_.inits.back(), init_state_manager_);

        // connect the fwds tasks
        this->edges(pipeline_state_manager_, layer_tasks_.fwds.front());
        for (size_t i = 0; i < layer_tasks_.fwds.size() - 1; ++i) {
            this->edges(layer_tasks_.fwds[i], layer_tasks_.fwds[i + 1]);
        }
        this->edges(layer_tasks_.fwds.back(), pipeline_state_manager_);

        // connect loss and bwds tasks
        if (loss_task_) {
            this->edges(pipeline_state_manager_, loss_task_);
            this->edges(loss_task_, layer_tasks_.bwds.back());
            for (size_t i = layer_tasks_.bwds.size() - 1; i >= 1; --i) {
                this->edges(layer_tasks_.bwds[i], layer_tasks_.bwds[i - 1]);
            }
            this->edges(init_state_manager_, loss_task_);
            this->edges(loss_task_, init_state_manager_);
            init_state_->has_loss = true;
        }

        // connect optimizer
        if (optimizer_task_) {
            optimizer_state_->nb_layers(layer_tasks_.layer_count);
            for (size_t i = 0; i < layer_tasks_.bwds.size(); ++i) {
                this->edges(layer_tasks_.bwds[i], optimizer_task_);
            }
            this->edges(optimizer_task_, optimizer_state_manager_);
            this->edges(optimizer_state_manager_, pipeline_state_manager_);
        }
    }

  public:
    /*
     * Create the NNState with allocated parameters and the gradient for the
     * network if needed. The rest of the data required for the computation is
     * allocated in `init_state`.
     */
    std::shared_ptr<NNState<ftype>> create_state() {
        auto state = std::make_shared<NNState<ftype>>();

        this->pushData(std::make_shared<CreateParameterData<ftype>>(state));
        (void)this->get<CreateParameterData<ftype>>();
        this->cleanGraph();
        return state;
    }

    /*
     * Initialize data required for the computation. The parameters are not
     * allocated here, but all the tensors used for the computation (input,
     * output, error, temporary tensor, ...) are allocated in this function.
     *
     * This function can be used multiple times, and all the data is reallocated
     * each time. The function should be used whenever the batch_size is
     * changed (because this requires reallocation). Note that once this
     * function is called, all the tensors and tensor descriptors are properly
     * allocated and initialized, meaning that no allocation or initialization
     * will be done during the computation to ensure maximum performance.
     */
    void init_state(std::shared_ptr<NNState<ftype>> state,
                    tensor_dims_t input_dims) {
        this->pushData(std::make_shared<InitData<ftype>>(state, input_dims));
        (void)this->get<InitData<ftype>>();
        this->cleanGraph();
    }

    Tensor<ftype> const &predict(std::shared_ptr<NNState<ftype>> state,
                                 Tensor<ftype> const &input) {
        this->pushData(std::make_shared<PredictionData<ftype>>(state, &input));
        Tensor<ftype> const *output = this->get<PredictionData<ftype>>()->input;
        this->cleanGraph();
        return *output;
    }

    std::shared_ptr<NNState<ftype>> train(std::shared_ptr<NNState<ftype>> state,
                                          DataSet<ftype> const &ds,
                                          size_t epochs) {
        this->pushData(
            std::make_shared<TrainingData<ftype>>(state, ds, epochs));
        (void)this->get<TrainingData<ftype>>();
        this->cleanGraph();
        return state;
    }

  public:
    void terminate() {
        pipeline_state_->terminate();
        init_state_->terminate();
        this->finishPushingData();
        this->waitForTermination();
    }

  public:
    template <typename OutType> std::shared_ptr<OutType> get() {
        return std::get<std::shared_ptr<OutType>>(*this->getBlockingResult());
    }

  private:
    std::shared_ptr<InitState> init_state_ = nullptr;
    std::shared_ptr<InitStateManager> init_state_manager_ = nullptr;
    std::shared_ptr<PipelineState> pipeline_state_ = nullptr;
    std::shared_ptr<PipelineStateManager> pipeline_state_manager_ = nullptr;
    std::shared_ptr<LossTask> loss_task_ = nullptr;
    std::shared_ptr<OptimizerTask> optimizer_task_ = nullptr;
    std::shared_ptr<OptimizerState> optimizer_state_ = nullptr;
    std::shared_ptr<OptimizerStateManager> optimizer_state_manager_ = nullptr;
    LayerTasks layer_tasks_;
    cuda_data_t cuda_data_;
};

#endif
