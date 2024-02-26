/*
 * Copyright © 2024 MOQI SINGAPORE PTE. LTD.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 */

#pragma once
#include <span>
#include <variant>
#include <SearchIndex/Common/IndexDataIO.h>
#include <SearchIndex/Common/IndexSourceData.h>
#include <SearchIndex/SearchIndexCommon.h>
#include <faiss/IndexScalarQuantizer.h>

#include <stack>
#include <unordered_map>
#include "BlockingQueue.h"

namespace Search
{

struct PrefetchInfo
{
    // actual object is IOUringReadBuffer*, use void* to avoid link error
    void * read_buffer;
    std::vector<idx_t> prefetch_id_list;
    // number of data prefetched and read
    std::unordered_map<size_t, int> page_to_read_idx;
};

class MemFpBlock
{
public:
    explicit MemFpBlock(int id_) : id(id_) { }
    int id;
    void setDim(int dim_) { this->dim = dim_; }
    // used by decode
    void initializeDecodeBlock(const std::vector<idx_t> & ids)
    {
        int item_size = sizeof(float);
        auto new_size = dim * ids.size() * item_size;
        checkAvailableMemory(new_size);
        buffer.resize(new_size);
        id2addr.clear();
        int current = 0;
        for (const auto vec_id : ids)
        {
            auto * fp32_ptr = buffer.data() + dim * current * item_size;
            id2addr[vec_id] = fp32_ptr;
            current++;
        }
    }

    inline void * getDataPtr(idx_t vec_id) { return id2addr[vec_id]; }
    int dim;
    std::unordered_map<idx_t, void *> id2addr;
    std::vector<uint8_t> buffer;
};

class MemFp32Block : public MemFpBlock
{
public:
    void decodeFp16(
        uint8_t * begin_ptr, const faiss::IndexScalarQuantizer & quantizer)
    {
        for (auto & id_and_addr : id2addr)
        {
            const uint8_t * current_fp16
                = begin_ptr + id_and_addr.first * sizeof(uint16_t) * this->dim;
            quantizer.sa_decode(
                1, current_fp16, reinterpret_cast<float *>(id_and_addr.second));
        }
    }
    explicit MemFp32Block(int id_) : MemFpBlock(id_) { }
};

using MemId = int;

class MemoryManager
{
public:
    virtual ~MemoryManager() { }

    explicit MemoryManager(int num_blocks) : mem_queue(num_blocks)
    {
        for (int i = 0; i < num_blocks; i++)
        {
            auto block = std::make_unique<MemFp32Block>(i);
            fp32_blocks.emplace_back(std::move(block));
            mem_queue.put(i);
        }
    }

    std::vector<std::unique_ptr<MemFp32Block>> fp32_blocks;
    BlockingQueue<MemId> mem_queue;

    MemFp32Block * getFp32Block()
    {
        MemId take_id = 0;
        mem_queue.take(take_id);
        return fp32_blocks[take_id].get();
    }

    void releaseFp32Block(const MemFp32Block * block)
    {
        mem_queue.put(block->id);
    }
};

static std::unique_ptr<MemoryManager> mem_manager;
static std::mutex create_mem_manager_mutex;
static MemoryManager * getMemoryManager()
{
    std::lock_guard<std::mutex> lock(create_mem_manager_mutex);
    if (!mem_manager)
    {
        mem_manager = std::make_unique<MemoryManager>(16);
    }
    return mem_manager.get();
}

/// @brief Provides abstraction for in-memory or on disk data
template <typename T>
class DenseDataLayer
{
public:
    using DataChunk = DataSet<T>;
    // We use variant to achieve polymorphism and avoid modifying scann too much
    //   (e.g. adding too many template parameters)
    using IStreamPtr = std::variant<std::ifstream *, AbstractIStream *>;
    using OStreamPtr = std::variant<std::ofstream *, AbstractOStream *>;

    DenseDataLayer(size_t max_data_, size_t data_dim_) :
        max_data(max_data_),
        data_dim(data_dim_) {}

    // Add chunk of data to DataLayer
    void addData(std::shared_ptr<DataChunk> chunk)
    {
        SI_THROW_IF_NOT_MSG(
            this->data_num + chunk->numData() <= this->max_data,
            ErrorCode::LOGICAL_ERROR,
            "DenseDataLayer exceeding initialization size");
        SI_THROW_IF_NOT(
            this->data_dim == chunk->dimension(), ErrorCode::LOGICAL_ERROR);
        addDataImpl(*chunk);
    }

    size_t dataSize() const { return data_dim * sizeof(T); }

    // call when finish adding all the data
    virtual void seal() { }

    size_t dataNum() const { return data_num; }

    size_t dataDimension() const { return data_dim; }

    virtual const T * getDataPtr(idx_t idx) const = 0;

    virtual T * getDataPtr(idx_t idx) = 0;

    /* use variant type here to avoid adding template parameters */

    // load data_layer from the reader
    virtual void load(IStreamPtr reader, size_t data_num = -1) = 0;

    // serialize data_layer and return a checksum
    virtual size_t serialize(OStreamPtr writer) = 0;

    // remove and free up associated resources (e.g. memory & disk file)
    virtual void remove() = 0;

    virtual ~DenseDataLayer() { }

protected:
    inline thread_local static PrefetchInfo * thread_prefetch_info;
    inline thread_local static std::vector<char> thread_vec_decode_info;
    virtual void addDataImpl(DataChunk & chunk) = 0;


    size_t max_data{0};
    size_t data_num{0};
    size_t data_dim{0};
};

/** Compute distance with a subset of vectors
 *
 * @param data_layer    data layer to compute distance
 * @param x             query vectors, size n * d
 * @param labels        indices of the vectors that should be compared
 *                      for each query vector, size n * k
 * @param distances     corresponding output distances, size n * k
 */
void ComputeDistanceSubset(
    DenseDataLayer<float> * data_layer,
    Metric metric,
    const float * x,
    idx_t n,
    const std::span<idx_t> labels,
    float * distances);

}
