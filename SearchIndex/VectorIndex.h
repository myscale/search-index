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
#include <fstream>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <SearchIndex/Common/DenseBitmap.h>
#include <SearchIndex/SearchIndex.h>

namespace Search
{

/// @brief Base class for all vector indices.
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class VectorIndex : public SearchIndex<IS, OS, IDS, dataType>
{
public:
    using T = typename DataTypeMap<dataType>::type;
    using DistanceType = typename DataTypeMap<dataType>::distance_type;
    using DataSetPtr = std::shared_ptr<DataSet<T>>;
    using SharedLock = std::shared_lock<std::shared_mutex>;

    /// @brief Parameter for estimate filter ratio.
    static const int FILTER_SAMPLE_NUM = 10009;

    /// @brief Read 100MB data in one chunk at most during `add()`.
    static const size_t DEFAULT_TRAIN_DATA_CHUNK_SIZE = 100 * 1024 * 1024;

    /// @brief Read 4MB data in one chunk at most during `add()`
    /// to react more timely to abort().
    static const size_t DEFAULT_ADD_DATA_CHUNK_SIZE = 4 * 1024 * 1024;

    /// @brief Read & write to disk in 4MB blocks.
    static const size_t DISK_RW_BLOCK_SIZE = 4 * 1024 * 1024;

    // @brief bits per byte
    static const size_t CHAR_BITS = 8;

    VectorIndex(
        const std::string & name_,
        IndexType index_type_,
        Metric metric_,
        size_t data_dim_,
        size_t max_points_) :
        SearchIndex<IS, OS, IDS, dataType>(name_, max_points_),
        index_type(index_type_),
        metric(metric_),
        data_dim(data_dim_)
    {
        SI_LOG_INFO(
            "Create VectorIndex, IndexType {}, metric {}, data_dim {}, "
            "max_points {}",
            enumToString(index_type_),
            enumToString(metric_),
            data_dim_,
            max_points_);
    }

    /**
     * @brief Perform single or batch vector search.
     * @param queries Query vectors.
     * @param topK Number of nearest neighbors to search for.
     * @param params Search parameters.
     * @param first_stage_only Whether to perform first stage search only.
     * @param filter Filter to apply to the search.
     * @param stats QueryStats object pointer to store query statistics (skip if null).
     * @return SearchResult Search reuslt including randk and distance.
    */
    std::shared_ptr<SearchResult> search(
        DataSetPtr queries,
        int32_t topK,
        Parameters params,
        bool first_stage_only = false,
        IDS * filter = nullptr,
        QueryStats * stats = nullptr)
    {
        SharedLock lock(this->mutation_mutex);
        this->checkAndAbort();
        this->adaptSearchParams(params);
        SI_THROW_IF_NOT_MSG(
            queries->dimension() == this->data_dim,
            ErrorCode::LOGICAL_ERROR,
            "search vector dimension don't match");
        if (metric == Metric::Cosine)
            queries = queries->normalize();
        RECORD_MEMORY_USAGE("before search");
        auto res = searchImpl(
            queries, topK, params, first_stage_only, filter, stats);
        RECORD_MEMORY_USAGE("after search");
        for (size_t i = 0; i < res->numQueries(); ++i)
        {
            // transform from internal IP similarity to cosine distance
            if (metric == Metric::Cosine)
                for (auto & elem : res->getResultDistances(i))
                    elem = 1 - elem;
            // return data_ids for final stage search
            if (!first_stage_only)
                this->translateDataID(
                    res->getResultIndices(i).data(), res->getResultLength(i));
        }

        return res;
    }

    /// @brief Total maximum bytes of original data vectors.
    size_t getMaxDataBytes() const
    {
        if constexpr (dataType == DataType::BinaryVector)
        {
            return data_dim * this->maxDataPoints() / 8;
        }
        else
        {
            return sizeof(T) * data_dim * this->maxDataPoints();
        }
    }

    /// @brief Dimension of data vectors.
    size_t dataDimension() const
    {
        return data_dim;
    }

    /// @brief Byte size of each data vector.
    size_t vectorSize()
    {
        return this->data_dim * DataTypeMap<dataType>::bits / 8;
    }

    /// @brief Set chunk size for `train()`.
    void setTrainDataChunkSize(size_t train_data_chunk_size_)
    {
        train_data_chunk_size = train_data_chunk_size_;
    }

    /// @brief Set chunk size for `add()`.
    void setAddDataChunkSize(size_t add_data_chunk_size_)
    {
        add_data_chunk_size = add_data_chunk_size_;
    }

    /// @brief Whehter the vector index supports two stage search.
    virtual bool supportTwoStageSearch() const
    {
        return false;
    }

    /**
     * @brief Compute number of candidates for first stage search.
     *
     * If `num_data` of the index is unknown, pass -1 by default.
     * If `data_dim` of the index is unknown, pass -1 by default.
    */
    static int computeFirstStageNumCandidates(
        IndexType index_type,
        bool disk_mode,
        int64_t num_data,
        int64_t data_dim,
        int32_t topK,
        Parameters params);

    /// @brief Compute top-k distances for each query vector in the subset
    virtual std::shared_ptr<SearchResult> computeTopDistanceSubset(
        DataSetPtr /* queries */,
        std::shared_ptr<SearchResult> /* first_stage_result */,
        int32_t /* top_k */) const
    {
        SI_THROW_MSG(
            ErrorCode::NOT_IMPLEMENTED,
            "computeTopDistanceSubset() not implemented for generic "
            "VectorIndex!");
    }

protected:
    /// @brief Actual implementation of top-k search.
    virtual std::shared_ptr<SearchResult> searchImpl(
        DataSetPtr & queries,
        int32_t topK,
        Parameters & params,
        bool first_stage_only,
        IDS * filter,
        QueryStats * stats)
        = 0;

    /// @brief Estimate filter ratio for tuning filter search parameter.
    float estimateFilterRatio(IDS * filter)
    {
        static thread_local std::minstd_rand random_engine{2023};
        size_t num_tests = 0;
        size_t num_selected = 0;
        size_t num_nonzeros = 0;
        std::uniform_int_distribution<size_t> distribution(
            0, this->numData() - 1);

        for (idx_t i = 0; i < FILTER_SAMPLE_NUM; ++i)
        {
            idx_t pos = distribution(random_engine);
            auto idx = this->id_list ? this->id_list->at(pos) : pos;
            auto count_tests = filter->unsafe_approx_count_member(idx);
            num_tests += count_tests.first;
            num_selected += count_tests.second;
            num_nonzeros += count_tests.second > 0;
            // early stop when statistically signifcant
            if (num_selected >= 20 && num_nonzeros >= 6)
                break;
        }
        return static_cast<float>(num_selected) / num_tests;
    }

    IndexType index_type;
    Metric metric;
    size_t data_dim;
    size_t train_data_chunk_size = DEFAULT_TRAIN_DATA_CHUNK_SIZE;
    size_t add_data_chunk_size = DEFAULT_ADD_DATA_CHUNK_SIZE;
};

}
