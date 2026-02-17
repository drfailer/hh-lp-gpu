#ifndef TASK_INIT_TASK
#define TASK_INIT_TASK
#include "../data/init_data.hpp"
#include "../data/init_parameters_data.hpp"
#include "../model/layer/layer.hpp"
#include "../types.hpp"
#include "cuda_task.hpp"
#include <hedgehog/hedgehog.h>
#include <memory>

#define InitTaskIn                                                             \
    InitParametersData<ftype, InitTarget::Layer>,                              \
        InitData<ftype, InitTarget::Layer>
#define InitTaskOut                                                            \
    InitParametersData<ftype, InitTarget::Layer>,                              \
        InitData<ftype, InitTarget::Layer>
#define InitTaskIO 2, InitTaskIn, InitTaskOut

class InitTask : public CUDATask<InitTaskIO> {
  public:
    InitTask() : CUDATask<InitTaskIO>("InitTask") {}

    void execute(std::shared_ptr<InitParametersData<ftype, InitTarget::Layer>>
                     data) override {
        for (auto &layer : layers_) {
            LayerData<ftype> ld;
            auto param_shape = layer->parameters_shape();
            ld.w = tensor::tensor<ftype>(param_shape.w);
            ld.b = tensor::tensor<ftype>(param_shape.b);
            layer->init_parameters(cuda_, {ld.w, ld.b});
            data->states->layers_datas[layer->idx] = std::move(ld);
        }
        this->addResult(data);
    }

    void
    execute(std::shared_ptr<InitData<ftype, InitTarget::Layer>> data) override {
        auto dims = data->input_dims;
        auto &nn = data->network_data;

        // TODO: do not allocate gradients during the inference
        for (auto layer : layers_) {
            assert(layer->idx < nn->layers_datas.size());
            auto io_shape = layer->io_shape(dims);
            auto &ld = nn->layers_datas[layer->idx];

            // input dims of the next layer
            dims = io_shape.y.dims;

            // fwd init
            ld.x = tensor::tensor_view<ftype>(io_shape.x, nullptr);
            ld.y = tensor::tensor<ftype>(io_shape.y);
            layer->init_fwd(cuda_, ld);

            // bwd init
            ld.dx = tensor::tensor<ftype>(io_shape.x);
            ld.dy = tensor::tensor_view<ftype>(io_shape.y, nullptr);
            ld.dw = tensor::tensor_like<ftype>(ld.w);
            ld.db = tensor::tensor_like<ftype>(ld.b);
            layer->init_bwd(cuda_, ld);
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

    tensor::TensorShape input_shape(std::shared_ptr<NetworkData<ftype>> nn) const {
        assert(layers_.front()->idx < nn->layers_datas.size());
        assert(nn->layers_datas[layers_.front()->idx].x.size() > 0 && "this tensor should not be used on this rank");
        return nn->layers_datas[layers_.front()->idx].x.shape();
    }

    tensor::TensorShape output_shape(std::shared_ptr<NetworkData<ftype>> nn) const {
        assert(layers_.back()->idx < nn->layers_datas.size());
        assert(nn->layers_datas[layers_.back()->idx].dy.size() > 0 && "this tensor should not be used on this rank");
        return nn->layers_datas[layers_.back()->idx].dy.shape();
    }


  private:
    std::vector<std::shared_ptr<Layer<ftype>>> layers_ = {};
};

#endif
