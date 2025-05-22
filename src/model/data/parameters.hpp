#ifndef MODEL_DATA_PARAMETERS
#define MODEL_DATA_PARAMETERS

template <typename T>
struct parameter_t {
    T *weights = nullptr;
    T *biases = nullptr;
};

#endif
