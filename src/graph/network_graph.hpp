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
    InitParametersData<ftype>, InitData<ftype>, PredictionData<ftype>,         \
        TrainingData<ftype>
#define NetworkGraphOut                                                        \
    InitParametersData<ftype>, InitData<ftype>, PredictionData<ftype>,         \
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

    virtual ~NetworkGraph() {
        CUDNN_CHECK(cudnnDestroy(cuda_data_.cudnn_handle));
    }

  public:
    template <typename LayerType, typename... Types>
    void add_layer(Types... args) {
        layer_tasks_.add_layer(
            std::make_shared<LayerType>(std::forward<Types>(args)...));
    }

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
    virtual void build() {
        // connect the init tasks
        this->edges(init_state_manager_, layer_tasks_.inits.front());
        this->edges(layer_tasks_.inits.back(), init_state_manager_);

        // connect the fwds tasks
        this->edges(pipeline_state_manager_, layer_tasks_.fwds.front());
        this->edges(layer_tasks_.fwds.back(), pipeline_state_manager_);

        // connect loss and bwds tasks
        if (loss_task_) {
            this->edges(pipeline_state_manager_, loss_task_);
            this->edges(loss_task_, layer_tasks_.bwds.back());
            this->edges(init_state_manager_, loss_task_);
            this->edges(loss_task_, init_state_manager_);
            init_state_->has_loss = true;
        }

        // connect optimizer
        if (optimizer_task_) {
            optimizer_state_->nb_layers(layer_tasks_.layer_count);
            this->edges(layer_tasks_.bwds.front(), optimizer_task_);
            this->edges(optimizer_task_, optimizer_state_manager_);
            this->edges(optimizer_state_manager_, pipeline_state_manager_);
        }
    }

  public:
    /*
     * Create the NetworkData with allocated parameters and the gradient for the
     * network if needed. The rest of the data required for the computation is
     * allocated in `init_state`.
     */
    virtual std::shared_ptr<NetworkData<ftype>> init_parameters() {
        auto nn = std::make_shared<NetworkData<ftype>>(this->layer_tasks_.layer_count);

        this->pushData(std::make_shared<InitParametersData<ftype>>(nn));
        (void)this->get<InitParametersData<ftype>>();
        this->cleanGraph();
        return nn;
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
    virtual void init(std::shared_ptr<NetworkData<ftype>> nn,
              tensor::dims_t input_dims) {
        this->pushData(std::make_shared<InitData<ftype>>(nn, input_dims));
        (void)this->get<InitData<ftype>>();
        this->cleanGraph();
    }

    // TODO: the data set system will be changed to allow taking const input!

    virtual tensor::Tensor const *predict(std::shared_ptr<NetworkData<ftype>> nn,
                                         tensor::Tensor &input) {
        this->pushData(std::make_shared<PredictionData<ftype>>(nn, &input));
        tensor::Tensor *output = this->get<PredictionData<ftype>>()->input;
        this->cleanGraph();
        return output;
    }

    virtual std::shared_ptr<NetworkData<ftype>>
    train(std::shared_ptr<NetworkData<ftype>> nn, DataSet<ftype> &ds, size_t epochs) {
        this->pushData(std::make_shared<TrainingData<ftype>>(nn, ds, epochs));
        (void)this->get<TrainingData<ftype>>();
        this->cleanGraph();
        return nn;
    }

  public:
    virtual void terminate() {
        pipeline_state_->terminate();
        init_state_->terminate();
        this->finishPushingData();
        this->waitForTermination();
    }

  public:
    template <typename OutType> std::shared_ptr<OutType> get() {
        return std::get<std::shared_ptr<OutType>>(*this->getBlockingResult());
    }

  protected:
    std::shared_ptr<InitState> init_state_ = nullptr;
    std::shared_ptr<InitStateManager> init_state_manager_ = nullptr;
    std::shared_ptr<PipelineState> pipeline_state_ = nullptr;
    std::shared_ptr<PipelineStateManager> pipeline_state_manager_ = nullptr;
    std::shared_ptr<LossTask> loss_task_ = nullptr;
    std::shared_ptr<OptimizerTask> optimizer_task_ = nullptr;
    std::shared_ptr<OptimizerState> optimizer_state_ = nullptr;
    std::shared_ptr<OptimizerStateManager> optimizer_state_manager_ = nullptr;
    LayerTasks layer_tasks_;
    CUDA cuda_data_;
};

#endif
