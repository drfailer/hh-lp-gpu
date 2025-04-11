#ifndef TASK_LAYER_TASK_H
#define TASK_LAYER_TASK_H
#include "../data/bwd_data.hpp"
#include "../data/fwd_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LayerTaskIn FwdData<ftype>, BwdInputData<ftype>
#define LayerTaskOut FwdData<ftype>, BwdOutputData<ftype>
#define LayerTaskType 2, LayerTaskIn, LayerTaskOut

class LayerTask : public hh::AbstractCUDATask<LayerTaskType> {
  public:
    LayerTask(std::string const &name, size_t layer_idx)
        : hh::AbstractCUDATask<LayerTaskType>(name, 1),
          layer_idx_(layer_idx) {}

  public:
    size_t layer_idx() const { return layer_idx_; }

  private:
    size_t layer_idx_;
};

#endif
