#ifndef TASK_FWD_TASK_H
#define TASK_FWD_TASK_H
#include "../data/fwd_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>
#include <stdexcept>

#define FwdTaskIn FwdData<ftype>
#define FwdTaskOut FwdData<ftype>
#define FwdTaskIO 1, FwdTaskIn, FwdTaskOut

class FwdTask : public hh::AbstractCUDATask<FwdTaskIO> {
  public:
    FwdTask() : hh::AbstractCUDATask<FwdTaskIO>("FwdTask", 1) {}

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

    void execute(std::shared_ptr<FwdData<ftype>> data) override {
        auto *input = data->input;
        auto states = data->states;

        for (auto layer : layers_) {
            input = layer->fwd(cuda_data_, states->layers[layer->idx], input);
            CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        }
        data->input = input;
        this->addResult(data);
    }

    void add_layer(std::shared_ptr<Layer<ftype>> layer) {
        layers_.push_back(layer);
    }

    std::shared_ptr<hh::AbstractTask<FwdTaskIO>> copy() override {
        throw std::logic_error("error: FwdTask should not be copied.");
    }

    std::vector<std::shared_ptr<Layer<ftype>>> const &layers() const {
        return layers_;
    }

  private:
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
    cuda_data_t cuda_data_;
};

#endif
