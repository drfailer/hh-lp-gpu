#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>
#include <vector>

struct FwdData {
    std::shared_ptr<NetworkData> network_data;
    tensor::Tensor *input;

    // NOTE:
    // this is completely unoptimize and a lot of memory is reallocated during
    // the training, however, this is only requried because the current MPI
    // version installed on the test machine was not compiled with cuda.
    static inline std::vector<char> transfer_buffer = std::vector<char>(1000000);

    hh::comm::Package pack() {
        assert(this->input != nullptr);
        assert(this->input->size() > 0 && this->input->data() != nullptr);
        size_t buffer_size = this->input->size() * this->input->element_size();
        if (this->transfer_buffer.size() != buffer_size) {
            this->transfer_buffer.resize(buffer_size);
        }
        this->input->to_host(this->transfer_buffer.data());
        // printf("FwdData::pack(%ld)\n", this->input->size());
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    this->transfer_buffer.data(), buffer_size,
                },
            },
        };
    }
    void unpack(hh::comm::Package) {
        assert(this->input != nullptr);
        assert(this->input->size() > 0 && this->input->data() != nullptr);
        assert(this->transfer_buffer.data() != nullptr);
        // printf("FwdData::unpack(%ld)\n", this->input->size());
        this->input->from_host(this->transfer_buffer.data());
    }
    hh::comm::Package package() {
        assert(this->input != nullptr);
        assert(this->input->size() > 0 && this->input->data() != nullptr);
        size_t buffer_size = this->input->size() * this->input->element_size();
        if (this->transfer_buffer.size() != buffer_size) {
            this->transfer_buffer.resize(buffer_size);
        }
        // printf("FwdData::package(%ld)\n", this->input->size());
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    this->transfer_buffer.data(), buffer_size,
                }
            },
        };
    }
};

#endif
