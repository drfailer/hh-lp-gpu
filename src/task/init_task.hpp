#ifndef TASK_INIT_TASK
#define TASK_INIT_TASK
#include "../data/init_parameters_data.hpp"
#include "../data/init_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>
#include <memory>

#define InitTaskIn                                                             \
    InitParametersData<ftype, InitTarget::Layer>,                              \
        InitData<ftype, InitTarget::Layer>
#define InitTaskOut                                                            \
    InitParametersData<ftype, InitTarget::Layer>,                              \
        InitData<ftype, InitTarget::Layer>
#define InitTaskIO 2, InitTaskIn, InitTaskOut

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

    void execute(std::shared_ptr<InitParametersData<ftype, InitTarget::Layer>>
                     data) override {
        for (auto &layer : layers_) {
            LayerData<ftype> ld;
            auto param_shape = layer->parameters_shape();
            ld.w = tensor::tensor<ftype>(param_shape.w);
            ld.b = tensor::tensor<ftype>(param_shape.b);
            layer->init_parameters(cuda_data_, {ld.w, ld.b});
            data->states->layers.push_back(ld);
        }
        this->addResult(data);
    }

    void
    execute(std::shared_ptr<InitData<ftype, InitTarget::Layer>> data) override {
        auto dims = data->input_dims;
        auto &states = data->states;

        // TODO: do not allocate gradients during the inference
        for (auto layer : layers_) {
            auto io_shape = layer->io_shape(dims);
            auto &ld = states->layers[layer->idx];

            // input dims of the next layer
            dims = io_shape.y.dims;

            // fwd init
            ld.x = tensor::tensor_view<const ftype>(io_shape.x, nullptr);
            ld.y = tensor::tensor<ftype>(io_shape.y);
            layer->init_fwd(cuda_data_, ld);

            // bwd init
            ld.dx = tensor::tensor<ftype>(io_shape.x);
            ld.dy = tensor::tensor_view<const ftype>(io_shape.y, nullptr);
            ld.dw = tensor::tensor_like<ftype>(ld.w);
            ld.db = tensor::tensor_like<ftype>(ld.b);
            layer->init_bwd(cuda_data_, ld);
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
