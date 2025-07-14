#ifndef TASK_INIT_TASK
#define TASK_INIT_TASK
#include "../data/create_parameter_data.hpp"
#include "../data/init_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>
#include <memory>

#define InitTaskIn                                                             \
    CreateParameterData<ftype, CreateParameterTarget::Layer>,                  \
        InitData<ftype, InitTarget::Layer>
#define InitTaskOut                                                            \
    CreateParameterData<ftype, CreateParameterTarget::Layer>,                  \
        InitData<ftype, InitTarget::Layer>
#define InitTaskIO 2, InitTaskIn, InitTaskOut

// We might want to have only one LayerTask that does everything (we don't need
// parallelisation in this task).

class InitTask : public hh::AbstractCUDATask<InitTaskIO> {
  public:
    InitTask() : hh::AbstractCUDATask<InitTaskIO>("InitTask") {}

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

    void execute(std::shared_ptr<
                 CreateParameterData<ftype, CreateParameterTarget::Layer>>
                     data) override {
        for (auto &layer : layers_) {
            data->states->layers.emplace_back(layer->create_parameters());
            // TODO: this should not be done here v
            data->states->layers.back().create_gradient_tensors();
        }
        this->addResult(data);
    }

    void
    execute(std::shared_ptr<InitData<ftype, InitTarget::Layer>> data) override {
        auto dims = data->input_dims;
        auto &states = data->states;

        for (auto layer : layers_) {
            dims = layer->init(cuda_data_, states->layers[layer->idx], dims);
        }
        data->input_dims = dims;
        this->addResult(data);
    }

    void add_layer(std::shared_ptr<Layer<ftype>> layer) {
        layers_.push_back(layer);
    }

    std::shared_ptr<hh::AbstractTask<InitTaskIO>> copy() override {
        throw std::logic_error("error: InitTask should not be copied.");
    }

  private:
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
    cuda_data_t cuda_data_;
};

#endif
