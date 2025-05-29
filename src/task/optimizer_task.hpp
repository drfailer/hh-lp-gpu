#ifndef TASK_OPTIMIZER_OPTIMIZER_TASK_H
#define TASK_OPTIMIZER_OPTIMIZER_TASK_H
#include "../data/opt_layer_data.hpp"
#include "../model/optimizer/optimizer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define OptimizerTaskIn OptLayerData<ftype>
#define OptimizerTaskOut OptLayerData<ftype>
#define OptimizerTaskIO 1, OptimizerTaskIn, OptimizerTaskOut

class OptimizerTask : public hh::AbstractCUDATask<OptimizerTaskIO> {
  public:
    using OptimizerList = std::vector<std::shared_ptr<Optimizer<ftype>>>;

  public:
    OptimizerTask(size_t nb_threads)
        : hh::AbstractCUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizers_(std::make_shared<OptimizerList>()) {}

    OptimizerTask(std::shared_ptr<OptimizerList> optimizers, size_t nb_threads)
        : hh::AbstractCUDATask<OptimizerTaskIO>("Optimizer", nb_threads),
          optimizers_(optimizers) {}

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

    void execute(std::shared_ptr<OptLayerData<ftype>> data) override {
        optimizers_->operator[](data->idx)->optimize(
            cuda_data_, data->state->layers[data->idx]);
        CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        this->addResult(data);
    }

    std::shared_ptr<hh::AbstractTask<OptimizerTaskIO>> copy() override {
        return std::make_shared<OptimizerTask>(optimizers_,
                                               this->numberThreads());
    }

    void add_layer(std::shared_ptr<Optimizer<ftype>> layer_optimizer) {
        optimizers_->push_back(layer_optimizer);
    }

  private:
    std::shared_ptr<OptimizerList> optimizers_ = nullptr;
    cuda_data_t cuda_data_;
};

#endif
