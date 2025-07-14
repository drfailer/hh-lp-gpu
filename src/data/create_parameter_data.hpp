#ifndef DATA_CREATE_PARAMETER_DATA
#define DATA_CREATE_PARAMETER_DATA
#include "../model/data/nn_state.hpp"
#include <memory>

enum class CreateParameterTarget {
    Network,
    Layer,
};

template <typename T,
          CreateParameterTarget target = CreateParameterTarget::Network>
struct CreateParameterData {
    std::shared_ptr<NNState<T>> states;
};

#endif
