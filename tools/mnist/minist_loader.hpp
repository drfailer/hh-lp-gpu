#ifndef MNIST_MINIST_LOADER_H
#define MNIST_MINIST_LOADER_H
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

  auto load_imgages(std::string const &path) {
    ifstream_type fs(path, std::ios::binary);
    [[maybe_unused]] unsigned int magic = 0, size = 0, rows = 0, cols = 0;

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

    std::vector<Tensor<ftype> *> images(size);

    for (size_t i = 0; i < size; ++i) {
      std::vector<ftype> image(rows * cols);
      for (size_t px = 0; px < image.size(); ++px) {
        unsigned char px_value;
        fs.get(reinterpret_cast<byte &>(px_value));
        assert(0 <= px_value && px_value <= 255);
        image[px] = (ftype)px_value / 255.;
      }
      auto *image_gpu = new Tensor<ftype>({1, 1, rows * cols, 1},
                                          {rows * cols, rows * cols, 1, 1});
      CUDA_CHECK(
          memcpy_host_to_gpu(image_gpu->data(), image.data(), rows * cols));
      cudaDeviceSynchronize();
      images[i] = image_gpu;
    }

    return images;
  }

  auto create_output_vector(int label) {
    auto *result_gpu = new Tensor<ftype>({1, 1, 10, 1}, {10, 10, 1, 1});
    std::vector<ftype> result_host(10, 0);

    result_host[label] = 1;
    CUDA_CHECK(memcpy_host_to_gpu(result_gpu->data(), result_host.data(), 10));

    return result_gpu;
  }

  DataSet<ftype> load_ds(std::string const labels_path,
                         std::string const images_path) {
    auto labels = load_labels(labels_path);
    auto images = load_imgages(images_path);

    DataSet<ftype> ds;

    for (size_t i = 0; i < images.size(); ++i) {
      ds.datas.emplace_back(images[i], create_output_vector(labels[i]));
    }
    return ds;
  }
};

#endif
