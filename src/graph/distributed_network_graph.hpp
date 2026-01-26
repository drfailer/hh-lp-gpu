#ifndef GRAPH_DISTRIBUTED_NETWORK_GRAPH
#define GRAPH_DISTRIBUTED_NETWORK_GRAPH
#include "network_graph.hpp"
#include <memory>
#include <vector>
#include <hedgehog/communicator/communicator_task.hpp>

// TODO: to facilitate the data transmission, we should have intermediate tasks
//       that would be use to prepare the data. This way, we can use a global
//       bank to store tensors, and only transmit tensor IDs when needed. Const
//       data will also be an issue on the recver end (does the intermediate
//       tasks solve this?). NetworkData<T> is also an issue since the current
//       implementation contains the actual weights tensors which should never
//       be shared between processes.
//       Finally, we will need to use several memory pool that will each
//       contain one element. Therefore, I think that the implementation of the
//       memory pool should be changed to facilitate this. Having a callback
//       instead of the pool in the core would make things easier, however, we
//       cannot add another template parameter here which makes things
//       challenging.

class DistributedNetworkGraph : public NetworkGraph {
  public:
    DistributedNetworkGraph(hh::comm::CommService *service)
        : service_(service),
          fwd_comms_({std::make_shared<hh::CommunicatorTask<FwdData<ftype>>>(service)}),
          bwd_comms_({std::make_shared<hh::CommunicatorTask<BwdData<ftype>>>(service)}),
          init_comms_({std::make_shared<hh::CommunicatorTask<InitTaskIn>>(service)}),
          optimizer_comm_(std::make_shared<hh::CommunicatorTask<OptLayerData<ftype>>>(service)) {}

  public:
    void cut_layer() {
        this->layer_tasks_.cut_layer();
        this->fwd_comms_.push_back(
                std::make_shared<hh::CommunicatorTask<FwdData<ftype>>>(this->service_));
        this->bwd_comms_.push_back(
                std::make_shared<hh::CommunicatorTask<BwdData<ftype>>>(this->service_));
        this->init_comms_.push_back(
                std::make_shared<hh::CommunicatorTask<InitTaskIn>>(this->service_));
    }

    template <typename LayerType, typename... Types>
    void add_layer(Types... args) {
        layer_tasks_.add_layer(
                std::make_shared<LayerType>(std::forward<Types>(args)...));
    }

    void build() override {
        if (this->service_->nbProcesses() != this->layer_tasks_.inits.size()) {
            // TODO: properly format the error message and specify the numbers!
            std::logic_error("error: the number of processes is different from the number of cut layers.");
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
            this->edges(this->loss_task_, this->layer_tasks_.bwds.back());
            this->edges(this->init_state_manager_, this->loss_task_);
            this->edges(this->loss_task_, this->init_state_manager_);
            this->init_state_->has_loss = true;
        }

        // connect the bwds tasks
        for (size_t i = this->service_->nbProcesses() - 1; i > 0; --i) {
            this->edges(this->layer_tasks_.bwds.at(i), this->bwd_comms_.at(i));
            this->edges(this->bwd_comms_.at(i), this->layer_tasks_.bwds.at(i - 1));
        }

        // connect optimizer
        if (this->optimizer_task_) {
            this->optimizer_state_->nb_layers(this->layer_tasks_.layer_count);
            this->edges(this->layer_tasks_.bwds.at(this->service_->rank()), this->optimizer_task_);
            this->edges(this->optimizer_task_, this->optimizer_comm_);
            this->edges(this->optimizer_comm_, this->optimizer_state_manager_);
            this->edges(this->optimizer_state_manager_, this->pipeline_state_manager_);
        }
    }

  private:
    hh::comm::CommService *service_;
    std::vector<std::shared_ptr<hh::CommunicatorTask<FwdData<ftype>>>> fwd_comms_;
    std::vector<std::shared_ptr<hh::CommunicatorTask<BwdData<ftype>>>> bwd_comms_;
    std::vector<std::shared_ptr<hh::CommunicatorTask<InitTaskIn>>> init_comms_;
    std::shared_ptr<hh::CommunicatorTask<OptLayerData<ftype>>> optimizer_comm_;
    size_t cut_layer_idx_ = 0;
};

#endif
