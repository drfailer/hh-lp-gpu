#ifndef TASK_LAYER_TASKS
#define TASK_LAYER_TASKS
#include "bwd_task.hpp"
#include "fwd_task.hpp"
#include "init_task.hpp"
#include <hedgehog/hedgehog.h>

struct LayerTasks {
    std::vector<std::shared_ptr<FwdTask>> fwds = {};
    std::vector<std::shared_ptr<BwdTask>> bwds = {};
    std::vector<std::shared_ptr<InitTask>> inits = {};
    size_t layer_count = 0;

    LayerTasks()
        : fwds({std::make_shared<FwdTask>()}),
          bwds({std::make_shared<BwdTask>()}),
          inits({std::make_shared<InitTask>()}) {}

    void add_layer(std::shared_ptr<Layer> layer) {
        layer->idx = layer_count++;
        fwds.back()->add_layer(layer);
        bwds.back()->add_layer(layer);
        inits.back()->add_layer(layer);
    }

    void cut_layer() {
        if (fwds.back()->layers().size() == 0) {
            throw std::logic_error(
                "error: cannot add a cut layer after in an empty shard.");
        }
        fwds.push_back(std::make_shared<FwdTask>());
        bwds.push_back(std::make_shared<BwdTask>());
        inits.push_back(std::make_shared<InitTask>());
    }
};

#endif
