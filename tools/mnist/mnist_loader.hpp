#ifndef MNIST_MNIST_LOADER_H
#define MNIST_MNIST_LOADER_H
#include "../../src/data/data_set.hpp"
#include "../../src/tools/gpu.hpp"
#include "../../src/types.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// TODO: add a "batch_size" argument and create batch.
// WARN: start loading stuff on the cpu, suffle and then copy on the gpu (the
// data is sorted)

class MNISTLoader {
  public:
    using byte = char;
    using ifstream_type = std::basic_ifstream<byte>;

  public:
    unsigned int read_big_endian_uint(ifstream_type &fs) {
        byte buff[4];
        fs.read(buff, 4);
        std::swap(buff[0], buff[3]);
        std::swap(buff[1], buff[2]);
        return *reinterpret_cast<uint *>(buff);
    }

    std::vector<int> load_labels(std::string const &path) {
        ifstream_type fs(path, std::ios::binary);
        [[maybe_unused]] unsigned int magic = 0, size = 0;

        if (!fs.is_open()) {
            std::cerr << "error: can't open label file " << path << std::endl;
            return {};
        }
        std::cout << "loading labels " << path << "..." << std::endl;

        magic = read_big_endian_uint(fs);
        size = read_big_endian_uint(fs);

        std::cout << "magic = " << magic << "; size = " << size << std::endl;

        std::vector<int> labels(size);
        for (size_t i = 0; i < size; ++i) {
            byte label;
            fs.read(&label, 1);
            labels[i] = int(label);
        }
        return labels;
    }

    auto load_imgages(std::string const &path, int64_t batch_size) {
        ifstream_type fs(path, std::ios::binary);
        unsigned int magic = 0, size = 0, rows = 0, cols = 0;
        unsigned char px_value;

        if (!fs.is_open()) {
            std::cerr << "error: can't open image file " << path << std::endl;
            exit(1);
        }
        std::cout << "loading images " << path << "..." << std::endl;

        magic = read_big_endian_uint(fs);
        size = read_big_endian_uint(fs);
        rows = read_big_endian_uint(fs);
        cols = read_big_endian_uint(fs);

        std::cout << "magic = " << magic << "; size = " << size << std::endl;
        std::cout << "row & cols = " << rows << "x" << cols << std::endl;

        size_t nb_batches = size / batch_size;
        assert(size % batch_size == 0);
        std::vector<Tensor<ftype> *> images(nb_batches);
        std::vector<ftype> batch_host(batch_size * rows * cols);

        for (size_t b = 0; b < nb_batches; ++b) {
            auto *batch_tensor =
                create_tensor<ftype>({batch_size, 1, rows * cols, 1});

            for (size_t i = 0; i < batch_size; ++i) {
                auto image = &batch_host[i * rows * cols];

                for (size_t px = 0; px < (rows * cols); ++px) {
                    fs.get(reinterpret_cast<byte &>(px_value));
                    assert(0 <= px_value && px_value <= 255);
                    image[px] = (ftype)px_value / 255.;
                }
            }
            batch_tensor->from_host(batch_host.data());
            cudaDeviceSynchronize();
            images[b] = batch_tensor;
        }

        return images;
    }

    auto create_output_tensor(int *label, int64_t batch_size) {
        auto *batch_gpu = create_tensor<ftype>({batch_size, 1, 10, 1});
        std::vector<ftype> batch_host(batch_size * 10, 0);

        for (size_t i = 0; i < batch_size; ++i) {
            batch_host[i * 10 + label[i]] = 1;
        }
        batch_gpu->from_host(batch_host.data());

        return batch_gpu;
    }

    DataSet<ftype> load_ds(std::string const labels_path,
                           std::string const images_path,
                           int64_t batch_size = 1) {
        auto labels = load_labels(labels_path);
        auto images = load_imgages(images_path, batch_size);

        DataSet<ftype> ds;

        for (size_t i = 0; i < images.size(); ++i) {
            ds.datas.emplace_back(
                images[i],
                create_output_tensor(&labels[i * batch_size], batch_size));
        }
        return ds;
    }
};

#endif
