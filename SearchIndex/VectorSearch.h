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

#include <SearchIndex/Config.h>
#include <SearchIndex/LocalDiskFileStore.h>
#include <SearchIndex/VectorIndex.h>
#ifdef ENABLE_FAISS
#    include <SearchIndex/FaissIndex.h>
#endif
#ifdef ENABLE_SCANN
#    include <SearchIndex/ScaNNIndex.h>
#endif
namespace Search
{

inline static const std::vector<IndexType> BINARY_VECTOR_INDEX_TEST_TYPES = {
    IndexType::BinaryFLAT,
    IndexType::BinaryIVF,
    IndexType::BinaryHNSW,
};

// db depend on this variable
inline static const std::vector<IndexType> FLOAT_VECTOR_INDEX_TEST_TYPES = {
#ifdef ENABLE_FAISS
    IndexType::IVFFLAT,
    IndexType::IVFPQ,
    IndexType::IVFSQ,
    IndexType::FLAT,
    IndexType::HNSWfastFLAT,
    IndexType::HNSWfastPQ,
    IndexType::HNSWfastSQ,
    IndexType::HNSWFLAT,
    IndexType::HNSWPQ,
    IndexType::HNSWSQ,
#endif
#ifdef ENABLE_SCANN
    IndexType::SCANN,
#endif
};


/**
 * \brief create VectorIndex index, a simple overload of createVectorIndex function
 * 
 * \param name name of the name of index
 * \param metric metric of index
 * \param data_dim data dim of index
 * \param max_points max points of index
 * \param params create index parameters
 * \param max_threads max threads
 * \param file_store_prefix file store prefix
 * \param is_termination_call server is termination call
 * 
 * \return shared pointer VectorIndex instance
*/
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::shared_ptr<VectorIndex<IS, OS, IDS, dataType>> createVectorIndex(
    const std::string & name,
    IndexType index_type,
    Metric metric,
    size_t data_dim,
    size_t max_points,
    const Parameters & params,
    [[maybe_unused]] const size_t max_threads,
    const std::string & file_store_prefix = "",
    [[maybe_unused]] std::function<bool()> is_termination_call = {})
{
    return createVectorIndex<IS, OS, IDS, dataType>(
        name,
        index_type,
        metric,
        data_dim,
        max_points,
        params,
        file_store_prefix,
        true,
        true,
        is_termination_call);
}

/**
 * \brief create VectorIndex index
 *
 * \param name name of the name of index
 * \param metric metric of index
 * \param data_dim data dim of index
 *
 * \return shared pointer VectorIndex instance
 */
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::shared_ptr<VectorIndex<IS, OS, IDS, dataType>> createVectorIndex(
    const std::string & name,
    IndexType index_type,
    Metric metric,
    size_t data_dim,
    size_t max_points,
    const Parameters & params,
    [[maybe_unused]] const std::string & file_store_prefix = "",
    [[maybe_unused]] bool use_file_checksum = false,
    [[maybe_unused]] bool manage_cache_folder = false,
    [[maybe_unused]] std::function<bool()> is_termination_call = {})
{
    switch (index_type)
    {
#ifdef ENABLE_FAISS
        case IndexType::FLAT:
        case IndexType::BinaryFLAT:
            return std::make_shared<FaissFlatIndex<IS, OS, IDS, dataType>>(
                name, index_type, metric, data_dim, max_points, params);
        case IndexType::IVFFLAT:
        case IndexType::IVFSQ:
        case IndexType::IVFPQ:
        case IndexType::BinaryIVF:
            return std::make_shared<FaissIVFIndex<IS, OS, IDS, dataType>>(
                name, index_type, metric, data_dim, max_points, params);
        case IndexType::HNSWfastFLAT:
        case IndexType::HNSWfastSQ:
        case IndexType::HNSWfastPQ:
        case IndexType::HNSWFLAT:
        case IndexType::HNSWSQ:
        case IndexType::HNSWPQ:
        case IndexType::BinaryHNSW:
            return std::make_shared<FaissHNSWIndex<IS, OS, IDS, dataType>>(
                name, index_type, metric, data_dim, max_points, params);
#endif
#ifdef ENABLE_SCANN
        case IndexType::SCANN: {
            if constexpr (dataType == DataType::FloatVector)
            {
                return std::make_shared<ScaNNIndex<IS, OS, IDS, dataType>>(
                    name, metric, data_dim, max_points, params);
            }
        }
#endif
        default:
            SI_THROW_MSG(
                ErrorCode::UNSUPPORTED_PARAMETER,
                "Unsupported IndexType: " + enumToString(index_type));
    }
}

/// Implement adaptive vector indexing algorithm with few lines of code
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::shared_ptr<VectorIndex<IS, OS, IDS, dataType>>
createFlatAdaptiveVectorIndex(
    const std::string & name,
    IndexType index_type,
    size_t flat_cutoff,
    Metric metric,
    size_t data_dim,
    size_t max_points,
    const Parameters & params)
{
#ifdef ENABLE_FAISS
    if (max_points <= flat_cutoff)
    {
        auto flat_index
            = std::make_shared<FaissFlatIndex<IS, OS, IDS, dataType>>(
                name + "_flat", metric, data_dim, max_points, params);
        // clear the search parameters for FlatIndex
        auto adapter
            = [](Parameters & search_params) { search_params.clear(); };
        flat_index->setSearchParamsAdapter(adapter);
    }
#endif
    return createVectorIndex<IS, OS, IDS, dataType>(
        name, index_type, metric, data_dim, max_points, params);
}

/// MyScale valid index parameters
extern const std::string MYSCALE_VALID_INDEX_PARAMETER;

std::string getDefaultIndexType(const DataType &search_type);
IndexType getVectorIndexType(std::string type, const DataType &search_type);
Metric getMetricType(std::string metric, const DataType &search_type);

}
