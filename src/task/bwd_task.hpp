#ifndef TASK_BWD_TASK_H
#define TASK_BWD_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/opt_layer_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include "cuda_task.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define BwdTaskIn BwdData<ftype>
#define BwdTaskOut BwdData<ftype>, OptLayerData<ftype>
#define BwdTaskIO 1, BwdTaskIn, BwdTaskOut

class BwdTask : public CUDATask<BwdTaskIO> {
  public:
    BwdTask() : CUDATask<BwdTaskIO>("BwdTask", 1) {}

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        auto *dy = data->error;
        auto &nn = data->network_data;

        for (int i = layers_.size() - 1; i >= 0; --i) {
            auto &ld = nn->layers_datas[layers_[i]->idx];
            ld.dy.data(dy->data());
            layers_[i]->bwd(cuda_, LayerBwdIn{ld.dy, ld.x, ld.y, ld.w, ld.b},
                            {ld.dx, ld.dw, ld.db});
            dy = &ld.dx;
            CUDA_CHECK(cudaStreamSynchronize(this->stream()));
            this->addResult(std::make_shared<OptLayerData<ftype>>(
                data->network_data, layers_[i]->idx));
        }
        data->error = dy;
        this->addResult(data);
    }

    void add_layer(std::shared_ptr<Layer<ftype>> layer) {
        layers_.push_back(layer);
    }

    std::shared_ptr<hh::AbstractTask<BwdTaskIO>> copy() override {
        throw std::logic_error("error: BwdTask should not be copied.");
    }

  private:
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
};

#endif
