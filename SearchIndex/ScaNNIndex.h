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
#include <iostream>
#include <sstream>
#include <SearchIndex/DataLayer.h>
#include <SearchIndex/IndexDataFileIO.h>
#include <SearchIndex/LocalDiskFileStore.h>
#include <SearchIndex/VectorIndex.h>

// forward declaration
namespace research_scann
{
template <typename T>
class SingleMachineSearcherBase;

class SingleMachineFactoryOptions;

class ScannConfig;

class ScannAsset;

class SearchParameters;

template <typename T>
class DenseDataset;
}

namespace Search
{

/**
 * @brief ScaNN Vector Index.
 */
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class ScaNNIndex final : public VectorIndex<IS, OS, IDS, dataType>
{
public:
    using T = typename DataTypeMap<dataType>::type;
    using DistanceType = typename DataTypeMap<dataType>::distance_type;
    using DataSetPtr = std::shared_ptr<DataSet<T>>;
    using ScannAsset = typename research_scann::ScannAsset;
    using DataShape = typename std::array<size_t, 2>;

    static const size_t VERION_MAJOR = 2;
    static const size_t VERION_MINOR = 0;
    static const size_t VERION_PATCH = 0;

    /// @brief Whether to use disk mode by default.
    static const int DEFAULT_DISK_MODE = 0;

    /// @brief Internal dimension must be a multiple of 8
    /// to avoid computing & prefetching issues.
    static const int SCANN_DIM_MULTIPLE = 8;

    /// Serialization file names, largely keep consistent with ScaNN for now.
    static inline const std::string SCANN_CONFIG_NAME = "scann_config.pb";
    static inline const std::string AH_CODEBOOK_NAME = "ah_codebook.pb";
    static inline const std::string PARTITIONER_NAME
        = "serialized_partitioner.pb";
    static inline const std::string TOKENIZATION_NAME
        = "datapoint_to_token.bin";
    static inline const std::string AH_DATASET_NAME = "hashed_dataset.bin";
    static inline const std::string AH_DATASET_BY_TOKEN_NAME
        = "hashed_dataset_by_token.bin";
    static inline const std::string AH_DATASET_PACKED_NAME
        = "hashed_dataset_packed.bin";
    static inline const std::string INT8_DATASET_NAME = "int8_dataset.bin";
    static inline const std::string INT8_MULTIPLIERS_NAME
        = "int8_multipliers.bin";
    static inline const std::string DATAPOINT_L2_NORM_NAME = "dp_norms.bin";
    static inline const std::string SCANN_ASSET_NAME = "scann_assets.pbtxt";


    /**
     * @brief filename of the dataset file.
     *
     * Handle serialization & loading original dataset specially: it takes a lot of space.
    */
    static inline const std::string DATASET_NAME = "dataset.bin";

    static inline const std::unordered_map<Metric, std::string>
        METRIC_TO_MEASURE_MAP = {
            {Metric::IP, "DotProductDistance"},
            {Metric::L2, "SquaredL2Distance"},
            {Metric::Cosine, "DotProductDistance"},
        };

    /// @brief Parameters for vector data quantization.
    struct HashParam
    {
        int clusters_per_block;
        std::string lookup_type;
        size_t hash_bits;
    };
    static inline const std::unordered_map<std::string, HashParam>
        HASH_TYPE_PARAMS
        = {{"lut16", {16, "INT8_LUT16", 4}}, {"lut256", {256, "INT8", 8}}};
    static const size_t BITS_PER_BYTE = 8;

    /// @brief Parameters for vector search with ScaNNIndex.
    struct SearchParams
    {
        float alpha;
        uint32_t num_reorder;
    };

    /**
     * @brief Construct a new ScaNNIndex object.
     *
     * @param name_ name of the index
     * @param metric_ distance metric used in the index
     * @param data_dim_ dimension of the data vectors
     * @param max_points_ maximum number of data vectors
     * @param params parameters for building the index
     * @param index_file_prefix_ prefix of the index file names
    */
    ScaNNIndex(
        const std::string & name_,
        Metric metric_,
        size_t data_dim_,
        size_t max_points_,
        Parameters params,
        std::string index_file_prefix_ = "");

    ~ScaNNIndex() override
    {
    }

    /**
     * @brief Get version of the index
     *
     * If `load_index_version` in parameters doesn't match `getVersion()`,
     * the constructor may throw an exception.
     */
    virtual IndexVersion getVersion() const override
    {
        return IndexVersion{VERION_MAJOR, VERION_MINOR, VERION_PATCH};
    }

    /// @brief Train centroids & quantization, no-op for ScaNNIndex.
    void train(
        IndexSourceDataReader<T> * /* reader */, int /* num_threads */) override
    {
    }

    /// @brief Add data to the index on the fly, not supported for ScaNNIndex for now.
    void
    add(IndexSourceDataReader<T> * /* reader */, int /* num_threads */) override
    {
        SI_THROW_MSG(
            ErrorCode::NOT_IMPLEMENTED, "ScaNN::add() not implemented yet!");
    }

    /// @brief Seal the index after `add()`, no-op for ScaNNIndex.
    void seal(int /* num_threads */) override
    {
        // nothing to do for ScaNNIndex
    }

    /// @brief load ScaNN index from IndexDataReader.
    void loadImpl(IndexDataReader<IS> * reader) override;

    /// @brief serialize ScaNN index with IndexDataWriter.
    void serializeImpl(IndexDataWriter<OS> * writer) override;

    /// @brief return the build progress of the index (between 0 and 1).
    float getBuildProgress() override
    {
        switch (this->status)
        {
            case IndexStatus::INIT:
                return 0;
            case IndexStatus::BUILDING:
                return 0.2f * this->numData() / this->max_points;
            case IndexStatus::SEALED:
            case IndexStatus::LOADED:
                return 1.0;
            case IndexStatus::LOADING:
                return NAN;
            default:
                SI_THROW_FMT(
                    ErrorCode::UNSUPPORTED_PARAMETER,
                    "Unknown IndexStatus: %s",
                    enumToString(this->status.load()).c_str());
        }
    }

    /// @brief return the memory & disk usage of the index.
    virtual IndexResourceUsage getResourceUsage() const final;

    /// @brief Extract search parameters from ScaNNIndex parameters.
    static SearchParams extractSearchParams(
        int32_t topK,
        Parameters & params,
        bool disk_mode,
        int64_t data_dim = -1,
        int64_t num_data = -1);

    /**
     * @brief Set the total number of additional threads for query pool.
     *
     * When `num_threads` are larger than 0, the query thread pool is set up
     * and will be used to reduce search latency. Especially useful for large
     * data part (e.g. larger than 20M) and ARM CPUs.
    */
    static void setQueryPoolThreads(int num_threads);

    /// @brief return the index file prefix in serialization and local cache files.
    std::string indexFilePrefix() const
    {
        return index_file_prefix;
    }

protected:
    /// @brief Search implementation for ScaNNIndex.
    std::shared_ptr<SearchResult> searchImpl(
        DataSetPtr & queries,
        int32_t topK,
        Parameters & params,
        bool first_stage_only,
        IDS * filter,
        QueryStats * stats) override;

    /// @brief Build the index from IndexSourceDataReader, with `num_threads` threads.
    void buildImpl(IndexSourceDataReader<T> * reader, int num_threads) override;

    /// @brief  Create data layer for storing data vectors.
    void createDataLayer()
    {
        data_layer = std::make_shared<DenseMemoryDataLayer<T>>(
            this->maxDataPoints(), this->scann_data_dim);
    }

    void
    createHashedDataLayerIfNeeded(size_t num_rows = 0, size_t hash_dim = 0);

    void createPackedHashedDataLayerIfNeeded(
        size_t packed_rows = 0, size_t hash_dim = 0);

    void buildScann(research_scann::SingleMachineFactoryOptions & opts);

    std::string getScannBuildConfigString();

    std::vector<research_scann::SearchParameters>
    getScannSearchParametersBatched(
        int batch_size,
        int final_nn,
        int pre_reorder_nn,
        int leaves,
        bool set_unspecified,
        IDS * filter) const;

    uint32_t getMaxNumLevels()
    {
        return num_children_per_level.size();
    }

    uint32_t getNumLeafNodes() const
    {
        // calculate the product of the number of children in each level
        uint32_t nodes = 1;
        for (auto c : num_children_per_level)
            nodes *= c;
        return nodes;
    }

    uint32_t getHashDim() const
    {
        return scann_data_dim / quantization_block_dimension
            * HASH_TYPE_PARAMS.at(hash_type).hash_bits / BITS_PER_BYTE;
    }


private:
    size_t scann_data_dim;
    std::vector<int32_t> num_children_per_level;

    /// @brief whether to build & store hashed data by token
    bool build_hashed_dataset_by_token;

    size_t min_cluster_size;
    size_t training_sample_size;
    size_t max_top_k;
    uint32_t quantization_block_dimension;
    float aq_threshold;
    std::string hash_type;
    bool residual_quantization;
    bool partition_random_init;
    bool quantize_centroids;
    bool spherical_partition;

    std::string index_file_prefix;
    std::shared_ptr<research_scann::ScannConfig> scann_config;
    std::shared_ptr<research_scann::SingleMachineSearcherBase<T>> scann;
    std::shared_ptr<DenseDataLayer<T>> data_layer;
    std::shared_ptr<DenseDataLayer<uint8_t>> hashed_data_by_token;
    std::shared_ptr<DenseDataLayer<uint8_t>> hashed_data_packed;
};

}
