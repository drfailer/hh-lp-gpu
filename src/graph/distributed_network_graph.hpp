#ifndef GRAPH_DISTRIBUTED_NETWORK_GRAPH
#define GRAPH_DISTRIBUTED_NETWORK_GRAPH
#include "network_graph.hpp"
#include "../tools/communicator.hpp"
#include <memory>
#include <vector>
#include <hedgehog/communicator/communicator_task.hpp>
#include "../tools/network_memory_manager.hpp"

#define SEND_TO(...) hh::comm::strategy::SendTo(__VA_ARGS__)

class DistributedNetworkGraph : public NetworkGraph {
  using InitParametersCommData = InitParametersData<ftype, InitTarget::Layer>;
  using InitCommData = InitData<ftype, InitTarget::Layer>;
  using FwdCommData = FwdData<ftype>;
  using BwdCommData = BwdData<ftype>;
  using InitComm = hh::CommunicatorTask<InitParametersCommData, InitCommData>;
  using FwdComm = hh::CommunicatorTask<FwdCommData>;
  using BwdComm = hh::CommunicatorTask<BwdCommData>;

  public:
    DistributedNetworkGraph(hh::comm::CommService *service)
        : service_(service),
          init_comms_({std::make_shared<InitComm>(service)}),
          fwd_comms_({std::make_shared<FwdComm>(service)}),
          bwd_comms_({std::make_shared<BwdComm>(service)}),
          optimizer_comm_(std::make_shared<hh::CommunicatorTask<OptLayerData<ftype>>>(service)) {
        // set the memory managers
        this->init_comms_.back()->setMemoryManager(&this->init_mm_);
        this->optimizer_comm_->setMemoryManager(&this->opt_mm_);

        // set send strategy
        hh::comm::rank_t dest = 1;
        this->init_comms_.back()->strategy<InitParametersCommData>(SEND_TO(dest));
        this->init_comms_.back()->strategy<InitCommData>(SEND_TO(dest));
        this->fwd_comms_.back()->strategy<FwdCommData>(SEND_TO(dest));
        this->bwd_comms_.back()->strategy<BwdCommData>(SEND_TO(0));
        this->optimizer_comm_->strategy<OptLayerData<ftype>>(SEND_TO(0));
    }

  public:
    void cut_layer() {
        this->layer_tasks_.cut_layer();

        // create the communicators
        this->init_comms_.push_back(std::make_shared<InitComm>(this->service_));
        this->fwd_comms_.push_back(std::make_shared<FwdComm>(this->service_));
        this->bwd_comms_.push_back(std::make_shared<BwdComm>(this->service_));

        // set the memory managers
        this->init_comms_.back()->setMemoryManager(&this->init_mm_);

        // set the strategies
        hh::comm::rank_t dest = this->init_comms_.size() % this->service_->nbProcesses();
        this->init_comms_.back()->strategy<InitParametersCommData>(SEND_TO(dest));
        this->init_comms_.back()->strategy<InitCommData>(SEND_TO(dest));
        this->fwd_comms_.back()->strategy<FwdCommData>(SEND_TO(dest));
        dest = this->init_comms_.size() - 1;
        this->bwd_comms_.back()->strategy<BwdCommData>(SEND_TO(dest));
    }

    template <typename LayerType, typename... Types>
    void add_layer(Types... args) {
        layer_tasks_.add_layer(
                std::make_shared<LayerType>(std::forward<Types>(args)...));
    }

    void build() override {
        if (this->service_->nbProcesses() != this->layer_tasks_.inits.size()) {
            std::ostringstream oss;
            oss << "error: " << this->service_->nbProcesses()
                << " processes created but the network has "
                << this->layer_tasks_.inits.size()
                << "cut layers (the number of processes must macth the number of cut layer).";
            throw std::logic_error(oss.str());
        }

        // connect the init tasks
        this->edges(this->init_state_manager_, this->layer_tasks_.inits.front());
        for (size_t i = 0; i < this->service_->nbProcesses() - 1; ++i) {
            this->edges(this->layer_tasks_.inits.at(i), this->init_comms_.at(i));
            this->edges(this->init_comms_.at(i), this->layer_tasks_.inits.at(i + 1));
        }
        this->edges(this->layer_tasks_.inits.back(), this->init_comms_.back());
        this->edges(this->init_comms_.back(), init_state_manager_);

        // connect the fwds tasks
        this->edges(this->pipeline_state_manager_, this->layer_tasks_.fwds.front());
        for (size_t i = 0; i < this->service_->nbProcesses() - 1; ++i) {
            this->edges(this->layer_tasks_.fwds.at(i), this->fwd_comms_.at(i));
            this->edges(this->fwd_comms_.at(i), this->layer_tasks_.fwds.at(i + 1));
        }
        this->edges(this->layer_tasks_.fwds.back(), this->fwd_comms_.back());
        this->edges(this->fwd_comms_.back(), pipeline_state_manager_);

        // connect loss and bwds tasks
        if (loss_task_) {
            this->edges(this->pipeline_state_manager_, this->loss_task_);
            this->edges(this->loss_task_, this->bwd_comms_.back());
            this->edges(this->init_state_manager_, this->loss_task_);
            this->edges(this->loss_task_, this->init_state_manager_);
            this->init_state_->has_loss = true;
        }

        // connect the bwds tasks
        for (size_t i = this->service_->nbProcesses() - 1; i > 0; --i) {
            this->edges(this->bwd_comms_.at(i), this->layer_tasks_.bwds.at(i));
            this->edges(this->layer_tasks_.bwds.at(i), this->bwd_comms_.at(i - 1));
        }
        this->edges(this->bwd_comms_.front(), this->layer_tasks_.bwds.front());

        // connect optimizer
        if (this->optimizer_task_) {
            this->optimizer_state_->nb_layers(this->layer_tasks_.layer_count);
            this->edges(this->layer_tasks_.bwds.at(this->service_->rank()), this->optimizer_task_);
            this->edges(this->optimizer_task_, this->optimizer_comm_);
            this->edges(this->optimizer_comm_, this->optimizer_state_manager_);
            this->edges(this->optimizer_state_manager_, this->pipeline_state_manager_);
        }

        this->service_->barrier();
    }

  public:
    std::shared_ptr<NetworkData<ftype>> init_parameters() override {
        auto nn = std::make_shared<NetworkData<ftype>>(this->layer_tasks_.layer_count);

        this->init_mm_.init(nn);
        this->service_->barrier();
        if (this->service_->rank() == 0) {
            this->pushData(std::make_shared<InitParametersData<ftype>>(nn));
            (void)this->get<InitParametersData<ftype>>();
        }
        this->service_->barrier();
        this->cleanGraph();
        return nn;
    }

    void init(std::shared_ptr<NetworkData<ftype>> nn, tensor::dims_t input_dims) override {
        std::shared_ptr<InitData<ftype>> init_data = nullptr;
        this->init_mm_.init(nn);
        this->service_->barrier();
        if (this->service_->rank() == 0) {
            this->pushData(std::make_shared<InitData<ftype>>(nn, input_dims));
            init_data = this->get<InitData<ftype>>();
        }
        this->service_->barrier();

        auto rank = this->service_->rank();

        // init and set memory managers for fwd tasks
        if (rank == 0) {
            this->fwd_mm_.init(nn, tensor::TensorShape(init_data->input_dims));
            this->fwd_comms_.back()->setMemoryManager(&this->fwd_mm_);
        } else {
            this->fwd_mm_.init(nn, this->layer_tasks_.inits[rank]->input_shape(nn));
            this->fwd_comms_[rank - 1]->setMemoryManager(&this->fwd_mm_);
        }
        this->fwd_comms_[rank]->setMemoryManager(&this->fwd_mm_);

        // init and set memory managers for bwd tasks
        this->bwd_mm_.init(nn, this->layer_tasks_.inits[rank]->output_shape(nn));
        if (rank == 0) {
            this->bwd_comms_.back()->setMemoryManager(&this->bwd_mm_);
        } else {
            this->bwd_comms_[rank - 1]->setMemoryManager(&this->bwd_mm_);
        }
        this->bwd_comms_[rank]->setMemoryManager(&this->bwd_mm_);

        // initializing the optimizer's memory manager
        this->opt_mm_.init(nn);

        this->service_->barrier();
        this->cleanGraph();
    }

    tensor::Tensor<ftype> const *predict(std::shared_ptr<NetworkData<ftype>> nn,
                                         tensor::Tensor<ftype> &input) override {
        tensor::Tensor<ftype> *output = nullptr;

        // this->service_->barrier();
        if (this->service_->rank() == 0) {
            this->pushData(std::make_shared<PredictionData<ftype>>(nn, &input));
            output = this->get<PredictionData<ftype>>()->input;
        }
        // this->service_->barrier();
        // this->cleanGraph();
        return output;
    }

    std::shared_ptr<NetworkData<ftype>>
    train(std::shared_ptr<NetworkData<ftype>> nn, DataSet<ftype> &ds, size_t epochs) override {
        this->service_->barrier();
        if (this->service_->rank() == 0) {
            this->pushData(std::make_shared<TrainingData<ftype>>(nn, ds, epochs));
            (void)this->get<TrainingData<ftype>>();
        }
        this->service_->barrier();
        this->cleanGraph();
        return nn;
    }


  private:
    hh::comm::CommService *service_;
    std::vector<std::shared_ptr<hh::CommunicatorTask<InitTaskIn>>> init_comms_;
    std::vector<std::shared_ptr<hh::CommunicatorTask<FwdData<ftype>>>> fwd_comms_;
    std::vector<std::shared_ptr<hh::CommunicatorTask<BwdData<ftype>>>> bwd_comms_;
    std::shared_ptr<hh::CommunicatorTask<OptLayerData<ftype>>> optimizer_comm_;
    size_t cut_layer_idx_ = 0;
    // memory managers
    InitMemoryManager init_mm_;
    FwdDataMemoryManager fwd_mm_;
    BwdDataMemoryManager bwd_mm_;
    OptLayerDataMemoryManager opt_mm_;
};

#undef SEND_TO

#endif
