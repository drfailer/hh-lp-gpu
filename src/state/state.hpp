#ifndef STATE_STATE
#define STATE_STATE

#define step_from_to(curr, to)                                                 \
    INFO_GRP("from stpe " #curr " to " #to ".", INFO_GRP_PIPELINE_STEP);       \
    if (step_ != curr) {                                                       \
        std::cerr << "error: entering step " #to " from step " #curr "."       \
                  << std::endl;                                                \
        return;                                                                \
    }                                                                          \
    step_ = to;

#endif
