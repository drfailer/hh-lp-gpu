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
          InitParametersData<InitTarget::Layer>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = InitParametersData<InitTarget::Layer>;

  public:
    InitParameterDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData> nn) {
        this->init_parameters_ = std::make_shared<ManagedType>(nn);
    }

    std::shared_ptr<ManagedType> allocate(AllocMode = AllocMode::Fail, LOC) override {
        // assert(this->init_parameters_ != nullptr);
        return this->init_parameters_;
    }

    void release(std::shared_ptr<ManagedType>&&, LOC) override {}

    std::string extraPrintingInformation() const override {
        return  "InitParametersData<InitTarget::Layer>";
    }

  private:
    std::shared_ptr<ManagedType> init_parameters_ = nullptr;
};

class InitDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<
          InitData<InitTarget::Layer>> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = InitData<InitTarget::Layer>;

  public:
    InitDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData> nn) {
        this->init_ = std::make_shared<ManagedType>();
        this->init_->network_data = nn;
    }

    std::shared_ptr<ManagedType> allocate(AllocMode = AllocMode::Fail, LOC) override {
        return init_;
    }

    void release(std::shared_ptr<ManagedType> &&, LOC) override {}

    std::string extraPrintingInformation() const override {
        return  "InitData<InitTarget::Layer>";
    }

  private:
    std::shared_ptr<ManagedType> init_ = nullptr;
};

class InitMemoryManager
    : public InitParameterDataMemoryManager,
      public InitDataMemoryManager {
  public:
    InitMemoryManager() = default;

    void init(std::shared_ptr<NetworkData> nn) {
        static_cast<InitParameterDataMemoryManager*>(this)->init(nn);
        static_cast<InitDataMemoryManager*>(this)->init(nn);
    }
};

/******************************************************************************/
/*                                    fwd                                     */
/******************************************************************************/

class FwdDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<FwdData> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = FwdData;

  public:
    FwdDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData> nn,
         tensor::TensorShape const          &input_shape) {
        this->input_tensor_ = tensor::tensor(input_shape);
        this->fwd_ = std::make_shared<ManagedType>(nn, &this->input_tensor_);
    }

    std::shared_ptr<ManagedType> allocate(AllocMode = AllocMode::Fail, LOC) override {
        if (this->fwd_ == nullptr) {
            return nullptr;
        }
        this->fwd_->input = &this->input_tensor_;
        return this->fwd_;
    }

    void release(std::shared_ptr<ManagedType> &&, LOC) override {}

  private:
    tensor::Tensor        input_tensor_ = {};
    std::shared_ptr<ManagedType> fwd_ = nullptr;
};

/******************************************************************************/
/*                                    bwd                                     */
/******************************************************************************/

class BwdDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<BwdData> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = BwdData;

  public:
    BwdDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData> nn,
         tensor::TensorShape const          &error_shape) {
        this->error_tensor_ = tensor::tensor(error_shape);
        this->bwd_ = std::make_shared<ManagedType>(nn, &this->error_tensor_);
    }

    std::shared_ptr<ManagedType> allocate(AllocMode = AllocMode::Fail, LOC) override {
        if (this->bwd_ == nullptr) {
            return nullptr;
        }
        this->bwd_->error = &this->error_tensor_;
        return this->bwd_;
    }

    void release(std::shared_ptr<ManagedType> &&, LOC) override {}

  private:
    tensor::Tensor        error_tensor_ = {};
    std::shared_ptr<ManagedType> bwd_ = nullptr;
};

/******************************************************************************/
/*                                 opt layer                                  */
/******************************************************************************/

class OptLayerDataMemoryManager
    : public hh::comm::tool::SingleTypeMemoryManager<OptLayerData> {
    using AllocMode = hh::comm::tool::MemoryManagerAllocateMode;
    using ManagedType = OptLayerData;

  public:
    OptLayerDataMemoryManager() = default;

    void init(std::shared_ptr<NetworkData> nn) {
        this->opt_ = std::make_shared<ManagedType>(nn, 0);
    }

    std::shared_ptr<ManagedType> allocate(AllocMode = AllocMode::Fail, LOC) override {
        if (this->opt_ == nullptr) {
            return nullptr;
        }
        this->opt_->idx = 0;
        return this->opt_;
    }

    void release(std::shared_ptr<ManagedType> &&, LOC) override {}

  private:
    std::shared_ptr<ManagedType> opt_ = nullptr;
};

#undef LOC

#endif
