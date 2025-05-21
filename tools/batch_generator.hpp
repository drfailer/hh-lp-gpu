#ifndef CUDNN_TOOLS_BATCH_GENERATOR
#define CUDNN_TOOLS_BATCH_GENERATOR
#include "../src/data/data_set.hpp"
#include <cassert>
#include <random>

template <typename T> class BatchGenerator {
public:
  BatchGenerator(int64_t seed) : rand(seed) {}

  DataSet<T> generate(DataSet<T> input_data, size_t input_size,
                      size_t ground_truth_size, size_t batch_count) {
    assert(batch_count > 1);
    DataSet<T> result;
    std::shuffle(input_data.datas.begin(), input_data.datas.end(), rand);

    for (size_t b = 0; b < input_data.datas.size() / batch_count; ++b) {
      Data<T> batch;
      size_t idx = b * batch_count;

      CUDA_CHECK(alloc_gpu(&batch.input, input_size * batch_count));
      CUDA_CHECK(
          alloc_gpu(&batch.ground_truth, ground_truth_size * batch_count));

      for (size_t i = 0; i < batch_count; ++i) {
        CUDA_CHECK(memcpy_gpu_to_gpu(&batch.input[i * input_size],
                                     input_data.datas[idx + i].input,
                                     input_size));
        CUDA_CHECK(memcpy_gpu_to_gpu(&batch.ground_truth[i * ground_truth_size],
                                     input_data.datas[idx + i].ground_truth,
                                     ground_truth_size));
      }
      result.datas.push_back(batch);
    }

    return result;
  }

private:
  std::minstd_rand rand;
};

#endif
