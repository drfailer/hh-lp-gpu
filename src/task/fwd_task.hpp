#ifndef TASK_FWD_TASK_H
#define TASK_FWD_TASK_H
#include "../data/fwd_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include "cuda_task.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define FwdTaskIn FwdData
#define FwdTaskOut FwdData
#define FwdTaskIO 1, FwdTaskIn, FwdTaskOut

class FwdTask : public CUDATask<FwdTaskIO> {
  public:
    FwdTask() : CUDATask<FwdTaskIO>("FwdTask", 1) {}

    void execute(std::shared_ptr<FwdData> data) override {
        tensor::Tensor *x = data->input;
        auto &nn = data->network_data;

        for (auto layer : layers_) {
            LayerData &ld = nn->layers_datas[layer->idx];
            ld.x.data(x->data());
            layer->fwd(cuda_, {ld.x, ld.w, ld.b}, {ld.y});
            x = &ld.y;
            CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        }
        data->input = x;
        this->addResult(data);
    }

    void add_layer(std::shared_ptr<Layer> layer) {
        layers_.push_back(layer);
    }

    std::shared_ptr<hh::AbstractTask<FwdTaskIO>> copy() override {
        throw std::logic_error("error: FwdTask should not be copied.");
    }

    std::vector<std::shared_ptr<Layer>> const &layers() const {
        return layers_;
    }

  private:
    std::vector<std::shared_ptr<Layer>> layers_ = {};
};

#endif
