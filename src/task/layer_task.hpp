#ifndef TASK_LAYER_TASK_H
#define TASK_LAYER_TASK_H
#include "../data/fwd_data.hpp"
#include "../data/bwd_data.hpp"
#include "../types.hpp"
#include <hedgehog/hedgehog.h>

#define LayerTaskIn FwdInputData<ftype>, BwdInputData<ftype>
#define LayerTaskOut FwdOutputData<ftype>, BwdOutputData<ftype>
#define LayerTaskType 2, LayerTaskIn, LayerTaskOut

struct LayerTask : hh::AbstractAtomicTask<LayerTaskType> {
    LayerTask(std::string const &name)
        : hh::AbstractAtomicTask<LayerTaskType>(name, 1) {}
};

#endif
