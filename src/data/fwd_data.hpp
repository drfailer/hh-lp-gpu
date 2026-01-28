#ifndef DATA_FWD_DATA_H
#define DATA_FWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>
#include <vector>

template <typename T> struct FwdData {
    std::shared_ptr<NetworkData<T>> network_data;
    tensor::Tensor<T> *input;

    // NOTE:
    // this is completely unoptimize and a lot of memory is reallocated during
    // the training, however, this is only requried because the current MPI
    // version installed on the test machine was not compiled with cuda.
    std::vector<T> transfer_buffer;

    hh::comm::Package pack() {
        assert(this->input != nullptr);
        assert(this->input->size() > 0 && this->input->data() != nullptr);
        if (this->transfer_buffer.size() != this->input->size()) {
            this->transfer_buffer.resize(this->input->size());
        }
        this->input->to_host(this->transfer_buffer.data());
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    (char*)this->transfer_buffer.data(), this->input->size() * sizeof(T),
                },
            },
        };
    }
    void unpack(hh::comm::Package) {
        assert(this->input != nullptr);
        assert(this->input->size() > 0 && this->input->data() != nullptr);
        assert(this->transfer_buffer.data() != nullptr);
        this->input->from_host(this->transfer_buffer.data());
    }
    hh::comm::Package package() {
        assert(this->input != nullptr);
        assert(this->input->size() > 0 && this->input->data() != nullptr);
        if (this->transfer_buffer.size() != this->input->size()) {
            this->transfer_buffer.resize(this->input->size());
        }
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    (char*)this->transfer_buffer.data(), this->input->size() * sizeof(T),
                }
            },
        };
    }
};

#endif
