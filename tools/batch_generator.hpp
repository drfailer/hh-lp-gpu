#ifndef CUDNN_TOOLS_BATCH_GENERATOR
#define CUDNN_TOOLS_BATCH_GENERATOR
#include "../src/data/data_set.hpp"
#include <cassert>
#include <random>

template <typename T> class BatchGenerator {
  public:
    BatchGenerator(int seed) : rand(seed) {}

    DataSet<T> generate(DataSet<T> input_data, int batch_size) {
        assert(batch_size > 1);
        DataSet<T> result;
        std::shuffle(input_data.datas.begin(), input_data.datas.end(), rand);

        for (size_t b = 0; b < input_data.datas.size() / batch_size; ++b) {
            Data<T> batch;
            auto batch_datas = &input_data.datas[b * batch_size];
            int c = input_data.datas[0].input->dims()[1];
            int h = input_data.datas[0].input->dims()[2];
            int w = input_data.datas[0].input->dims()[3];
            int gt_c = input_data.datas[0].ground_truth->dims()[1];
            int gt_h = input_data.datas[0].ground_truth->dims()[2];
            int gt_w = input_data.datas[0].ground_truth->dims()[3];

            batch.input = create_tensor<T>({batch_size, c, h, w});
            batch.ground_truth =
                create_tensor<T>({batch_size, gt_c, gt_h, gt_w});

            for (size_t i = 0; i < batch_size; ++i) {
                CUDA_CHECK(
                    memcpy_gpu_to_gpu(&batch.input->data()[i * c * h * w],
                                      batch_datas[i].input->data(), c * h * w));
                CUDA_CHECK(memcpy_gpu_to_gpu(
                    &batch.ground_truth->data()[i * gt_c * gt_h * gt_w],
                    batch_datas[i].ground_truth->data(), gt_c * gt_h * gt_w));
            }
            result.datas.push_back(batch);
        }

        return result;
    }

  private:
    std::minstd_rand rand;
};

#endif
