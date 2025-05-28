#ifndef TASK_BWD_TASK_H
#define TASK_BWD_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/opt_layer_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define BwdTaskIn BwdData<ftype>
#define BwdTaskOut BwdData<ftype>, OptLayerData<ftype>
#define BwdTaskIO 1, BwdTaskIn, BwdTaskOut

class BwdTask : public hh::AbstractCUDATask<BwdTaskIO> {
  public:
    BwdTask() : hh::AbstractCUDATask<BwdTaskIO>("BwdTask", 1) {}

    void initializeCuda() override {
        CUDNN_CHECK(cudnnCreate(&cuda_data_.cudnn_handle));
        CUDNN_CHECK(cudnnSetStream(cuda_data_.cudnn_handle, this->stream()));
        CUBLAS_CHECK(cublasCreate_v2(&cuda_data_.cublas_handle));
        CUBLAS_CHECK(
            cublasSetStream_v2(cuda_data_.cublas_handle, this->stream()));
    }

    void shutdownCuda() override {
        CUDNN_CHECK(cudnnDestroy(cuda_data_.cudnn_handle));
        CUBLAS_CHECK(cublasDestroy_v2(cuda_data_.cublas_handle));
    }

    void execute(std::shared_ptr<BwdData<ftype>> data) override {
        auto *error = data->error;
        auto &states = data->states;

        for (int i = layers_.size() - 1; i >= 0; --i) {
            error = layers_[i]->bwd(cuda_data_, states->layers[layers_[i]->idx],
                                    error);
            CUDA_CHECK(cudaStreamSynchronize(this->stream()));
            this->addResult(std::make_shared<OptLayerData<ftype>>(
                data->states, data->learning_rate, layers_[i]->idx));
        }
        data->error = error;
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
    cuda_data_t cuda_data_;
};

#endif
