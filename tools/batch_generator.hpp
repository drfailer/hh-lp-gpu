#ifndef CUDNN_TOOLS_BATCH_GENERATOR
#define CUDNN_TOOLS_BATCH_GENERATOR
#include "../src/data/data_set.hpp"
#include <cassert>
#include <random>

template <typename T> class BatchGenerator {
  public:
    BatchGenerator(int64_t seed) : rand(seed) {}

    DataSet<T> generate(DataSet<T> input_data, int64_t input_size,
                        int64_t ground_truth_size, int64_t batch_size) {
        assert(batch_size > 1);
        DataSet<T> result;
        std::shuffle(input_data.datas.begin(), input_data.datas.end(), rand);

        for (size_t b = 0; b < input_data.datas.size() / batch_size; ++b) {
            Data<T> batch;
            auto batch_datas = &input_data.datas[b * batch_size];

            batch.input = create_tensor<T>({batch_size, 1, input_size, 1});
            batch.ground_truth =
                create_tensor<T>({batch_size, 1, ground_truth_size, 1});

            for (size_t i = 0; i < batch_size; ++i) {
                CUDA_CHECK(memcpy_gpu_to_gpu(
                    &batch.input->data()[i * input_size],
                    batch_datas[i].input->data(), input_size));
                CUDA_CHECK(memcpy_gpu_to_gpu(
                    &batch.ground_truth->data()[i * ground_truth_size],
                    batch_datas[i].ground_truth->data(), ground_truth_size));
            }
            result.datas.push_back(batch);
        }

        return result;
    }

  private:
    std::minstd_rand rand;
};

#endif
