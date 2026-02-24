#ifndef DATA_BWD_DATA_H
#define DATA_BWD_DATA_H
#include "../model/data/network_data.hpp"
#include <memory>
#include <cstdio>

template <typename T> struct BwdData {
    std::shared_ptr<NetworkData<T>> network_data;
    tensor::Tensor *error;

    // NOTE:
    // this is completely unoptimize and a lot of memory is reallocated during
    // the training, however, this is only requried because the current MPI
    // version installed on the test machine was not compiled with cuda.
    static inline std::vector<T> transfer_buffer = std::vector<T>(1000000);

    hh::comm::Package pack() {
        assert(this->error != nullptr);
        assert(this->error->size() > 0 && this->error->data() != nullptr);
        if (this->transfer_buffer.size() != this->error->size()) {
            this->transfer_buffer.resize(this->error->size());
        }
        this->error->to_host(this->transfer_buffer.data());
        // printf("BwdData::pack(%ld)\n", this->error->size());
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    (char*)this->transfer_buffer.data(), this->error->size() * sizeof(T),
                },
            },
        };
    }
    void unpack(hh::comm::Package) {
        assert(this->error != nullptr);
        assert(this->error->size() > 0 && this->error->data() != nullptr);
        assert(this->transfer_buffer.data() != nullptr);
        // printf("BwdData::unpack(%ld)\n", this->error->size());
        this->error->from_host(this->transfer_buffer.data());
    }
    hh::comm::Package package() {
        assert(this->error != nullptr);
        assert(this->error->size() > 0 && this->error->data() != nullptr);
        if (this->transfer_buffer.size() != this->error->size()) {
            this->transfer_buffer.resize(this->error->size());
        }
        // printf("BwdData::package(%ld)\n", this->error->size());
        return hh::comm::Package{
            .data = {
                hh::comm::Buffer{
                    (char*)this->transfer_buffer.data(), this->error->size() * sizeof(T),
                }
            },
        };
    }
};

#endif
