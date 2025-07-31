#ifndef DATA_INIT_PARAMETERS_DATA
#define DATA_INIT_PARAMETERS_DATA
#include "../model/data/nn_state.hpp"
#include "init_data.hpp"
#include <memory>

template <typename T, InitTarget target = InitTarget::Network>
struct InitParametersData {
    std::shared_ptr<NNState<T>> states;
};

#endif
