#ifndef TOOLS_NETWORK_MEMORY_MANAGER
#define TOOLS_NETWORK_MEMORY_MANAGER
#include "communicator.hpp"
#include "../data/init_data.hpp"
#include "../data/init_parameters_data.hpp"
#include "../data/fwd_data.hpp"
#include "../data/bwd_data.hpp"
#include "../data/opt_layer_data.hpp"
#include <cassert>

// is undef at the end of this file
#define LOC std::source_location = std::source_location::current()

/******************************************************************************/
/*                                    init                                    */
/******************************************************************************/

class InitParameterDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<
          InitParametersData<ftype, InitTarget::Layer>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType
        = std::shared_ptr<InitParametersData<ftype, InitTarget::Layer>>;

  public:
    InitParameterDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData<ftype>> nn) {
        this->init_parameters_
            = std::make_shared<InitParametersData<ftype, InitTarget::Layer>>(
                nn);
    }

    ManagedType allocate(AllocMode = AllocMode::Fail, LOC) override {
        assert(this->init_parameters_ != nullptr);
        return this->init_parameters_;
    }

    void release(ManagedType &&, LOC) override {}

    std::string extraPrintingInformation() const override {
        return  "InitParametersData<ftype, InitTarget::Layer>";
    }

  private:
    ManagedType init_parameters_ = nullptr;
};

class InitDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<
          InitData<ftype, InitTarget::Layer>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = std::shared_ptr<InitData<ftype, InitTarget::Layer>>;

  public:
    InitDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData<ftype>> nn) {
        this->init_ = std::make_shared<InitData<ftype, InitTarget::Layer>>();
        this->init_->network_data = nn;
    }

    ManagedType allocate(AllocMode = AllocMode::Fail, LOC) override {
        return init_;
    }

    void release(ManagedType &&, LOC) override {}

    std::string extraPrintingInformation() const override {
        return  "InitData<ftype, InitTarget::Layer>";
    }

  private:
    ManagedType init_ = nullptr;
};

class InitMemoryManager
    : public InitParameterDataMemoryManager,
      public InitDataMemoryManager {
  public:
    InitMemoryManager() = default;

    void init(std::shared_ptr<NetworkData<ftype>> nn) {
        static_cast<InitParameterDataMemoryManager*>(this)->init(nn);
        static_cast<InitDataMemoryManager*>(this)->init(nn);
    }
};

/******************************************************************************/
/*                                    fwd                                     */
/******************************************************************************/

class FwdDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<FwdData<ftype>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = std::shared_ptr<FwdData<ftype>>;

  public:
    FwdDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData<ftype>> nn,
         tensor::TensorShape const          &input_shape) {
        this->input_tensor_ = tensor::Tensor<ftype>(input_shape);
        this->fwd_ = std::make_shared<FwdData<ftype>>(nn, &this->input_tensor_);
    }

    ManagedType allocate(AllocMode = AllocMode::Fail, LOC) override {
        this->fwd_->input = &this->input_tensor_;
        return this->fwd_;
    }

    void release(ManagedType &&, LOC) override {}

  private:
    tensor::Tensor<ftype> input_tensor_ = {};
    ManagedType           fwd_ = nullptr;
};

/******************************************************************************/
/*                                    bwd                                     */
/******************************************************************************/

class BwdDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<BwdData<ftype>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = std::shared_ptr<BwdData<ftype>>;

  public:
    BwdDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData<ftype>> nn,
         tensor::TensorShape const          &error_shape) {
        this->error_tensor_ = tensor::Tensor<ftype>(error_shape);
        this->bwd_ = std::make_shared<BwdData<ftype>>(nn, &this->error_tensor_);
    }

    ManagedType allocate(AllocMode = AllocMode::Fail, LOC) override {
        this->bwd_->error = &this->error_tensor_;
        return this->bwd_;
    }

    void release(ManagedType &&, LOC) override {}

  private:
    tensor::Tensor<ftype> error_tensor_ = {};
    ManagedType           bwd_ = nullptr;
};

/******************************************************************************/
/*                                 opt layer                                  */
/******************************************************************************/

class OptLayerDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<OptLayerData<ftype>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = std::shared_ptr<OptLayerData<ftype>>;

  public:
    OptLayerDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData<ftype>> nn) {
        this->nn_ = nn;
    }

    ManagedType allocate(AllocMode = AllocMode::Fail, LOC) override {
        return std::make_shared<OptLayerData<ftype>>(this->nn_, 0);
    }

    void release(ManagedType &&, LOC) override {}

  private:
    std::shared_ptr<NetworkData<ftype>> nn_;
};

#undef LOC

#endif
