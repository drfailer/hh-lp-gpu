#ifndef TASK_LOSS_TASK_H
#define TASK_LOSS_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/init_data.hpp"
#include "../data/loss_bwd_data.hpp"
#include "../data/loss_fwd_data.hpp"
#include "../model/loss/loss.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LossTaskIn                                                             \
    InitData<ftype, InitTarget::Loss>, LossFwdData<ftype>, LossBwdData<ftype>
#define LossTaskOut                                                            \
    InitData<ftype, InitTarget::Loss>, LossFwdData<ftype>, BwdData<ftype>
#define LossTaskIO 3, LossTaskIn, LossTaskOut

class LossTask : public hh::AbstractCUDATask<LossTaskIO> {
  public:
    LossTask(std::shared_ptr<Loss<ftype>> loss)
        : hh::AbstractCUDATask<LossTaskIO>("LossTask", 1), loss_(loss) {}

    void initializeCuda() override {
        CUDNN_CHECK(cudnnCreate(&cuda_.cudnn_handle));
        CUDNN_CHECK(cudnnSetStream(cuda_.cudnn_handle, this->stream()));
        CUBLAS_CHECK(cublasCreate_v2(&cuda_.cublas_handle));
        CUBLAS_CHECK(cublasSetStream_v2(cuda_.cublas_handle, this->stream()));
    }

    void shutdownCuda() override {
        CUDNN_CHECK(cudnnDestroy(cuda_.cudnn_handle));
        CUBLAS_CHECK(cublasDestroy_v2(cuda_.cublas_handle));
    }

    void
    execute(std::shared_ptr<InitData<ftype, InitTarget::Loss>> data) override {
        data->network_data->loss.tensor = tensor::tensor<ftype>(data->input_dims);
        this->addResult(data);
    }

    void execute(std::shared_ptr<LossFwdData<ftype>> data) override {
        loss_->fwd(cuda_, {*data->input}, {data->states->loss.tensor});
        CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        this->addResult(data);
    }

    void execute(std::shared_ptr<LossBwdData<ftype>> data) override {
        loss_->bwd(cuda_, {*data->y_true, *data->y_pred},
                   {data->states->loss.tensor});
        CUDA_CHECK(cudaStreamSynchronize(this->stream()));
        this->addResult(std::make_shared<BwdData<ftype>>(
            data->states, &data->states->loss.tensor));
    }

  private:
    std::shared_ptr<Loss<ftype>> loss_ = nullptr;
    CUDA cuda_;
};

#endif
