#ifndef TASK_LOSS_TASK_H
#define TASK_LOSS_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/loss_fwd_data.hpp"
#include "../model/loss/loss.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LossTaskIn LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskOut LossFwdData<ftype>, BwdData<ftype>
#define LossTaskIO 2, LossTaskIn, LossTaskOut

class LossTask : public hh::AbstractCUDATask<LossTaskIO> {
  public:
    LossTask(std::shared_ptr<Loss<ftype>> loss) : loss_(loss) {}

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

    void init(std::shared_ptr<NNState<ftype>> state) {
        auto model_output = state->layers.back().output;
        state->loss.tensor = create_tensor<ftype>(model_output->dims());
    }

    void execute(std::shared_ptr<LossFwdData<ftype>> data) override {
        loss_->fwd(cuda_data_, data->states->loss, data->input,
                   data->ground_truth);
        CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        this->addResult(data);
    }

    void execute(std::shared_ptr<LossBwdData<ftype>> data) override {
        loss_->bwd(cuda_data_, data->states->loss, data->input,
                   data->ground_truth);
        CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        this->addResult(std::make_shared<BwdData<ftype>>(
            data->states, data->states->loss.tensor, data->learning_rate));
    }

  private:
    std::shared_ptr<Loss<ftype>> loss_ = nullptr;
    cuda_data_t cuda_data_;
};

#endif
