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
#include <omp.h>
#include <SearchIndex/Common/DenseBitmap.h>
#include <SearchIndex/VectorIndex.h>
#include <faiss/Index.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexHNSWfast.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/MetricType.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace Search
{

static const size_t IVF_MIN_CLUSTER_SIZE = 50;

/// @brief IDSelector in faiss indices.
template <IDSelector IDS>
struct FaissIDSelector final : public faiss::IDSelector
{
    explicit FaissIDSelector(std::vector<idx_t> * id_list_, IDS * selector_) :
        id_list(id_list_), selector(selector_)
    {
    }

    /// Get the data_id, and check whether its in the selector
    bool is_member(idx_t pos) const override
    {
        return selector->is_member(id_list->at(pos));
    }

private:
    std::vector<idx_t> * id_list;
    IDS * selector;
};

/// @brief Index data writer for faiss indices.
template <typename OS>
struct FaissIOWriter final : public faiss::IOWriter
{
    FaissIOWriter(const char * name_, OS * out_stream_) :
        out_stream(out_stream_)
    {
        this->name = name_;
    }

    // fwrite. Return number of items written
    size_t operator()(const void * ptr, size_t size, size_t nitems) override
    {
        const auto * data = static_cast<const char *>(ptr);
        out_stream->write(data, size * nitems);
        // check stream status
        SI_THROW_IF_NOT_FMT(
            out_stream->good(),
            ErrorCode::CANNOT_WRITE_TO_OSTREAM,
            "writing to %s",
            name.c_str());
        return nitems;
    }

    // return a file number that can be memory-mapped
    int fileno() override
    {
        throw std::runtime_error("FaissIOWriter::fileno not implemented");
    }

private:
    OS * out_stream;
};

/// @brief Index data reader for faiss indices.
template <typename IS>
struct FaissIOReader final : public faiss::IOReader
{
    FaissIOReader(const char * name_, IS * in_stream_) : in_stream(in_stream_)
    {
        this->name = name_;
    }

    // fread. Returns number of items read or 0 in case of EOF.
    size_t operator()(void * ptr, size_t size, size_t nitems) override
    {
        auto * data = static_cast<char *>(ptr);
        in_stream->read(data, size * nitems);
        auto ret = in_stream->gcount();
        if (nitems && ret == 0)
        {
            return 0;
        }
        return ret / size;
    }

    // return a file number that can be memory-mapped
    int fileno() override
    {
        throw std::runtime_error("FaissIOReader::fileno not implemented");
    }

private:
    IS * in_stream;
};


inline faiss::MetricType getFaissMetric(Metric metric)
{
    switch (metric)
    {
        case Metric::L2:
            return faiss::METRIC_L2;
        case Metric::IP:
            return faiss::METRIC_INNER_PRODUCT;
        case Metric::Hamming:
            return faiss::METRIC_HAMMING;
        case Metric::Cosine:
            return faiss::METRIC_INNER_PRODUCT;
        case Metric::Jaccard:
            return faiss::METRIC_JACCARD;
    }
    // All cases has been covered, so we remove the default case to avoid
    // warning "default label in switch which covers all")
};

inline std::pair<faiss::ScalarQuantizer::QuantizerType, size_t>
parseScalarQuantizer(std::string bit_size)
{
    static std::unordered_map<
        std::string,
        std::pair<faiss::ScalarQuantizer::QuantizerType, size_t>>
        sq_map
        = {{"8bit", std::pair{faiss::ScalarQuantizer::QT_8bit, 8}},
           {"8bit_uniform",
            std::pair{faiss::ScalarQuantizer::QT_8bit_uniform, 8}},
           {"8bit_direct",
            std::pair{faiss::ScalarQuantizer::QT_8bit_direct, 8}},
           {"6bit", std::pair{faiss::ScalarQuantizer::QT_6bit, 6}},
           {"4bit", std::pair{faiss::ScalarQuantizer::QT_4bit, 4}},
           {"4bit_uniform",
            std::pair{faiss::ScalarQuantizer::QT_4bit_uniform, 4}},
           {"QT_fp16", std::pair{faiss::ScalarQuantizer::QT_fp16, 16}}};
    auto it = sq_map.find(bit_size);
    if (it == sq_map.end())
    {
        SI_THROW_MSG(
            ErrorCode::UNSUPPORTED_PARAMETER,
            "Unknown ScalarQuantization: " + bit_size);
    }
    return it->second;
};


/**
 * @brief Helper class for reading & writing float faiss indices.
 *
 * Template specialization is used to handle binary indices.
*/
template <DataType>
struct FaissDataTypeHelper
{
    // by default, it's faiss::Index (actually only works for float)
    using IndexType = faiss::Index;

    static IndexType * readIndex(faiss::IOReader * f, int io_flags = 0)
    {
        return faiss::read_index(f, io_flags);
    }

    static void writeIndex(IndexType * index, faiss::IOWriter * f)
    {
        faiss::write_index(index, f);
    }
};

/// @brief Helper class for reading & writing binary faiss indices.
template <>
struct FaissDataTypeHelper<DataType::BinaryVector>
{
    // use IndexBinary for BinaryVector data
    using IndexType = faiss::IndexBinary;

    static IndexType * readIndex(faiss::IOReader * f, int io_flags = 0)
    {
        return read_index_binary(f, io_flags);
    }

    static void writeIndex(IndexType * index, faiss::IOWriter * f)
    {
        faiss::write_index_binary(index, f);
    }
};


/**
 * @brief Common abstract class for Faiss indices.
 *
 * Because each family of vector indices (say IVF/HNSW) have completely different parameters,
 * it makes sense to have subclasses for these index algorithms.
 */
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class FaissIndex : public VectorIndex<IS, OS, IDS, dataType>
{
public:
    using IndexDataType = typename DataTypeMap<dataType>::type;
    using DataSetPtr = std::shared_ptr<DataSet<IndexDataType>>;
    using SearchParamsPtr = std::shared_ptr<faiss::SearchParameters>;
    using DistanceType = typename DataTypeMap<dataType>::distance_type;
    using FaissIndexPtr = typename FaissDataTypeHelper<dataType>::IndexType *;

    static const size_t VERION_MAJOR = 1;
    static const size_t VERION_MINOR = 0;
    static const size_t VERION_PATCH = 0;

    static inline const std::string FAISS_INDEX_DATA_BIN = "data_bin";

    static std::unordered_set<IndexType> binaryIndexTypes()
    {
        return {
            IndexType::BinaryIVF, IndexType::BinaryHNSW, IndexType::BinaryFLAT};
    }

    FaissIndex(
        const std::string & name_,
        IndexType index_type_,
        Metric metric_,
        size_t data_dim_,
        size_t max_points_) :
        VectorIndex<IS, OS, IDS, dataType>(
            name_, index_type_, metric_, data_dim_, max_points_)
    {
        faiss_metric = getFaissMetric(this->metric);

        if (binaryIndexTypes().contains(index_type_))
        {
            // parameter checks for binary indices
            SI_THROW_IF_NOT_MSG(
                this->data_dim % 8 == 0,
                ErrorCode::UNSUPPORTED_PARAMETER,
                "Binary index vector length must be a multiple of 8");
            SI_THROW_IF_NOT_MSG(
                metric_ == Metric::Hamming || metric_ == Metric::Jaccard,
                ErrorCode::UNSUPPORTED_PARAMETER,
                "Binary index metric must be Hamming or Jaccard for now");
            SI_THROW_IF_NOT_MSG(
                dataType == DataType::BinaryVector,
                ErrorCode::UNSUPPORTED_PARAMETER,
                "Binary index data type must be BinaryVector(uint8)");
        }
    }

    virtual IndexVersion getVersion() const override
    {
        return IndexVersion{VERION_MAJOR, VERION_MINOR, VERION_PATCH};
    }

    void buildImpl(
        IndexSourceDataReader<IndexDataType> * reader, int num_threads) override
    {
        const auto & usage = this->getResourceUsage();

        SI_LOG_INFO(
            "usage.memory_usage_bytes {}, disk_usage_bytes {}, "
            "build_memory_usage_bytes {}, build_disk_usage_bytes {}",
            usage.memory_usage_bytes,
            usage.disk_usage_bytes,
            usage.build_memory_usage_bytes,
            usage.build_disk_usage_bytes);
        setNumThreads(num_threads);
        // execute the following steps using default thread number
        train(reader, 0);
        this->checkAndAbort();
        if (dynamic_cast<faiss::IndexHNSWfast *>(index()))
        {
            dynamic_cast<faiss::IndexHNSWfast *>(index())->init_hnsw(
                VectorIndex<IS, OS, IDS, dataType>::max_points);
        }
        add(reader, 0);
        this->checkAndAbort();
        seal(0);
        this->status = IndexStatus::SEALED;
    }

    void train(
        IndexSourceDataReader<IndexDataType> * reader, int num_threads) override
    {
        this->status = IndexStatus::BUILD_TRAINING;
        if (this->index_type == IndexType::HNSWFLAT
            || this->index_type == IndexType::HNSWfastFLAT)
            // skip training for HNSWFLAT and HNSWfastFLAT and save IO
            return;

        if (this->index_type == IndexType::FLAT)
        {
            // for FLAT index, we need to train the index without data to reserve memory
            index()->train(0, nullptr);
            return;
        }

        setNumThreads(num_threads);
        // sample data for training
        size_t max_rows = this->train_data_chunk_size / this->vectorSize();
        auto samples = reader->sampleData(max_rows);
        SI_LOG_INFO(
            "{} start training, max_rows {}, sample data {}",
            this->getName(),
            max_rows,
            samples->numData());

        faiss::IndexBinary * binary
            = dynamic_cast<faiss::IndexBinary *>(index());
        if (binary)
        {
            binary->train(
                samples->numData(),
                reinterpret_cast<uint8_t *>(samples->getRawData()));
        }
        else
        {
            faiss::Index * float_index = dynamic_cast<faiss::Index *>(index());
            float_index->train(
                samples->numData(),
                reinterpret_cast<float *>(samples->getRawData()));
        }
    }

    void
    add(IndexSourceDataReader<IndexDataType> * reader, int num_threads) override
    {
        this->status = IndexStatus::BUILD_ADDING;

        SI_LOG_INFO("{} start adding vectors", this->getName());
        setNumThreads(num_threads);
        size_t max_rows = this->add_data_chunk_size / this->vectorSize();
        while (!reader->eof())
        {
            auto chunk = reader->readData(max_rows);
            if (VectorIndex<IS, OS, IDS, dataType>::metric == Metric::Cosine)
            {
                chunk = chunk->normalize(true);
            }
            faiss::IndexBinary * binary
                = dynamic_cast<faiss::IndexBinary *>(index());
            if (binary)
            {
                binary->add(
                    chunk->numData(),
                    reinterpret_cast<uint8_t *>(chunk->getRawData()));
            }
            else
            {
                faiss::Index * float_index
                    = dynamic_cast<faiss::Index *>(index());
                float_index->add(
                    chunk->numData(),
                    reinterpret_cast<float *>(chunk->getRawData()));
            }
            this->checkAndAbort();
        }
    }

    void seal(int /* num_threads */) override
    {
        // nothing to do here
    }

    /// @brief Return index build progress, between 0 and 1.
    virtual float getBuildProgress() override
    {
        if (this->status == IndexStatus::BUILD_ADDING)
        {
            return static_cast<float>(
                (20 + 80 * index()->ntotal / this->max_points) / 100.0);
        }
        // return default values in other cases
        return SearchIndex<IS, OS, IDS, dataType>::getBuildProgress();
    }

    /// @brief Algorithm specific parameters, to be implemented by subclasses.
    virtual SearchParamsPtr
    extractSearchParams(Parameters & params, IDS * filter)
        = 0;

    void loadImpl(IndexDataReader<IS> * reader) override
    {
        this->status = IndexStatus::LOADING;

        SI_LOG_INFO("{} starts loading", this->getName());
        auto in_stream = reader->getFieldDataInputStream(FAISS_INDEX_DATA_BIN);
        FaissIOReader faiss_reader("faiss_index_reader", in_stream.get());
        FaissIndexPtr index_load
            = FaissDataTypeHelper<dataType>::readIndex(&faiss_reader);

        // do some basic sanity_checking
        SI_THROW_IF_NOT(index()->d == index_load->d, ErrorCode::LOGICAL_ERROR);
        SI_THROW_IF_NOT(
            index()->metric_type == index_load->metric_type,
            ErrorCode::LOGICAL_ERROR);

        // set the smart pointer to the newly loaded
        resetFaissIndex(index_load);
        this->status = IndexStatus::LOADED;
    }

    void serializeImpl(IndexDataWriter<OS> * writer) override
    {
        SI_LOG_INFO("FaissIndex serialize");
        auto out_stream = writer->getFieldDataOutputStream(FAISS_INDEX_DATA_BIN);
        FaissIOWriter faiss_writer("faiss_index_writer", out_stream.get());
        FaissDataTypeHelper<dataType>::writeIndex(index(), &faiss_writer);
    }

    /**
     * @brief Extract and set index parameters.
     *
     * Must be called when base faiss index is constructed already.
     */
    virtual void extractAndSetCommonIndexParams(Parameters & params)
    {
        auto version_str
            = params.extractParam("load_index_version", std::string(""));
        if (!version_str.empty())
        {
            SI_LOG_INFO(
                "Creating {} index version {}",
                enumToString(this->index_type),
                version_str);
            auto version_loaded = IndexVersion::fromString(version_str);
            SI_THROW_IF_NOT_FMT(
                version_loaded == getVersion(),
                ErrorCode::UNSUPPORTED_PARAMETER,
                "Unsupported created version %s, current version %s",
                version_loaded.toString().c_str(),
                getVersion().toString().c_str());
        }

        size_t verbose = params.extractParam("verbose", 0);
        index()->verbose = static_cast<bool>(verbose);
    }

    std::shared_ptr<faiss::IndexBinary> getFaissBinaryIndex()
    {
        return index_binary;
    }

protected:
    std::shared_ptr<SearchResult> searchImpl(
        DataSetPtr & queries,
        int32_t topK,
        Parameters & params,
        bool /* first_stage_only */,
        IDS * filter,
        QueryStats * stats) override
    {
        // extract search parameters
        SearchParamsPtr search_params = extractSearchParams(params, filter);
        raiseErrorOnUnknownParams(params);
        // set the ID filter if it's not empty
        FaissIDSelector<IDS> faiss_selector(this->getDataID().get(), filter);
        search_params->sel = filter ? &faiss_selector : nullptr;
        search_params->stats = stats;

        auto res = SearchResult::createTopKHolder(queries->numData(), topK);
        // perform search

        faiss::IndexBinary * binary
            = dynamic_cast<faiss::IndexBinary *>(index());
        if (binary)
        {
            binary->search(
                queries->numData(),
                reinterpret_cast<uint8_t *>(queries->getRawData()),
                topK,
                reinterpret_cast<int32_t *>(res->getResultDistances()),
                res->getResultIndices(),
                search_params.get());
        }
        else
        {
            faiss::Index * float_index = dynamic_cast<faiss::Index *>(index());
            float_index->search(
                queries->numData(),
                reinterpret_cast<float *>(queries->getRawData()),
                topK,
                res->getResultDistances(),
                res->getResultIndices(),
                search_params.get());
        }

        return res;
    }

    void setNumThreads(int num_threads)
    {
        if (num_threads > 0)
        {
            omp_set_num_threads(static_cast<int>(num_threads));
            SI_LOG_INFO("Set num_threads={}", num_threads);
        }
    }

    /// @brief Set the internal faiss index field in constructor.
    void setFaissIndex(std::shared_ptr<faiss::Index> index)
    {
        index_float = index;
    }

    void setFaissIndex(std::shared_ptr<faiss::IndexBinary> index)
    {
        index_binary = index;
    }

    /// @brief Reset the internal faiss index pointer after loading.
    void resetFaissIndex(faiss::Index * index) { index_float.reset(index); }
    void resetFaissIndex(faiss::IndexBinary * index)
    {
        index_binary.reset(index);
    }

    /// @brief Return the corresponding faiss index according to data type
    /// (float or binary).
    FaissIndexPtr index()
    {
        if (dataType == DataType::BinaryVector)
        {
            return reinterpret_cast<FaissIndexPtr>(index_binary.get());
        }
        else
        {
            return reinterpret_cast<FaissIndexPtr>(index_float.get());
        }
    }

    faiss::MetricType faiss_metric;

private:
    // only one of these should be non-null according to data type
    std::shared_ptr<faiss::Index> index_float{nullptr};
    std::shared_ptr<faiss::IndexBinary> index_binary{nullptr};
};


/**
 * @brief IVF-family vector index.
 *
 * IVF index have completely different search parameters than other index types,
 * say HNSW, so it makes sense to have one class for it.
 */
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class FaissIVFIndex : public FaissIndex<IS, OS, IDS, dataType>
{
public:
    using SearchParamsPtr = std::shared_ptr<faiss::SearchParameters>;

    FaissIVFIndex(
        const std::string & name_,
        IndexType index_type_,
        Metric metric_,
        size_t data_dim_,
        size_t max_points_,
        Parameters params) :
        FaissIndex<IS, OS, IDS, dataType>(
            name_, index_type_, metric_, data_dim_, max_points_)
    {
        coarse_quantizer = std::make_shared<faiss::IndexFlat>(
            this->data_dim, 0, this->faiss_metric);
        // shared parameters for all IVF index types
        uint32_t ncentroids = params.extractParam("ncentroids", 1024);
        size_t max_clusters = this->max_points / IVF_MIN_CLUSTER_SIZE;
        size_t nlist = std::max(
            1UL, ncentroids > max_clusters ? max_clusters : ncentroids);
        SI_LOG_INFO(
            "ncentroids: {}, max_clusters: {}, nlist: {}",
            ncentroids,
            max_clusters,
            nlist);
        switch (this->index_type)
        {
            case IndexType::IVFSQ: {
                std::string bit_size
                    = params.extractParam("bit_size", std::string("8bit"));
                auto quantizer = parseScalarQuantizer(bit_size);
                this->setFaissIndex(
                    std::make_shared<faiss::IndexIVFScalarQuantizer>(
                        coarse_quantizer.get(),
                        this->data_dim,
                        nlist,
                        quantizer.first,
                        this->faiss_metric));
                SI_LOG_INFO(
                    "Create IVFSQ Index, nlist={} bit_size={}",
                    nlist,
                    bit_size);
                bits_per_dimension = static_cast<int>(quantizer.second);
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            }
            case IndexType::IVFPQ: {
                uint32_t M = params.extractParam("M", 32);
                uint32_t bit_size = params.extractParam("bit_size", 8);
                this->setFaissIndex(std::make_shared<faiss::IndexIVFPQ>(
                    coarse_quantizer.get(),
                    this->data_dim,
                    nlist,
                    M,
                    bit_size,
                    this->faiss_metric));
                SI_LOG_INFO(
                    "Create IVFPQ Index, nlist={} M={} bit_size={}",
                    nlist,
                    M,
                    bit_size);
                bits_per_dimension = bit_size;
                num_quantization_blocks = M;
                break;
            }
            case IndexType::IVFFLAT:
                this->setFaissIndex(std::make_shared<faiss::IndexIVFFlat>(
                    coarse_quantizer.get(),
                    this->data_dim,
                    nlist,
                    this->faiss_metric));
                SI_LOG_INFO("Create IVFFlat Index, nlist={}", nlist);
                bits_per_dimension = sizeof(float) * this->CHAR_BITS;
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            case IndexType::BinaryIVF:
                binary_coarse_quantizer
                    = std::make_shared<faiss::IndexBinaryFlat>(this->data_dim);
                this->setFaissIndex(std::make_shared<faiss::IndexBinaryIVF>(
                    binary_coarse_quantizer.get(),
                    this->data_dim,
                    nlist,
                    this->faiss_metric));
                SI_LOG_INFO("Create IVFFlat Index, nlist={}", nlist);
                bits_per_dimension = 1;
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            default:
                SI_THROW_MSG(
                    ErrorCode::UNSUPPORTED_PARAMETER,
                    "Unknown IVF Index Type: " + enumToString(index_type_));
        }
        this->extractAndSetCommonIndexParams(params);
        raiseErrorOnUnknownParams(params);
    }

    SearchParamsPtr
    extractSearchParams(Parameters & params, IDS * /*filter*/) override
    {
        size_t nprobe = params.extractParam("nprobe", 96);
        auto * ivf_params = new faiss::IVFSearchParameters;
        ivf_params->nprobe = nprobe;
        return SearchParamsPtr(ivf_params);
    }

    virtual IndexResourceUsage getResourceUsage() const final
    {
        size_t row_bytes = DIV_ROUND_UP(
            bits_per_dimension * num_quantization_blocks, this->CHAR_BITS);
        /// load memory usage and search memory usage is almost same
        auto theoretical_memory_usage = this->maxDataPoints() * row_bytes;
        float actual_build_memory_usage = theoretical_memory_usage * 1.33f;
        float load_memory_usage = theoretical_memory_usage * 1.14f;
        auto usage = IndexResourceUsage{
            .memory_usage_bytes = static_cast<size_t>(load_memory_usage),
            .disk_usage_bytes = 0,
            .build_memory_usage_bytes
            = static_cast<size_t>(actual_build_memory_usage),
            .build_disk_usage_bytes = 0};
        SI_LOG_INFO(
            "IVF getResourceUsage: {}MB", usage.memory_usage_bytes >> 20);
        return usage;
    }

private:
    int bits_per_dimension;
    int num_quantization_blocks;
    std::shared_ptr<faiss::IndexFlat> coarse_quantizer;
    std::shared_ptr<faiss::IndexBinaryFlat> binary_coarse_quantizer;
};


/// @brief Faiss HNSW index family.
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class FaissHNSWIndex : public FaissIndex<IS, OS, IDS, dataType>
{
public:
    using SearchParamsPtr = std::shared_ptr<faiss::SearchParameters>;
    using IndexHNSWPtr = std::shared_ptr<faiss::IndexHNSW>;
    using IndexHNSWfastPtr = std::shared_ptr<faiss::IndexHNSWfast>;
    using IndexBinaryHNSWPtr = std::shared_ptr<faiss::IndexBinaryHNSW>;

    FaissHNSWIndex(
        const std::string & name_,
        IndexType index_type_,
        Metric metric_,
        size_t data_dim_,
        size_t max_points_,
        Parameters params) :
        FaissIndex<IS, OS, IDS, dataType>(
            name_, index_type_, metric_, data_dim_, max_points_)
    {
        m = params.extractParam("m", 64);
        ef_c = params.extractParam("ef_c", 200);
        IndexHNSWPtr hnsw_index = nullptr;
        IndexBinaryHNSWPtr binary_hnsw_index = nullptr;
        IndexHNSWfastPtr hnswfast_index = nullptr;

        switch (this->index_type)
        {
            case IndexType::HNSWSQ:
            case IndexType::HNSWfastSQ: {
                std::string bit_size
                    = params.extractParam("bit_size", std::string("8bit"));
                auto quantizer = parseScalarQuantizer(bit_size);
                if (this->index_type == IndexType::HNSWSQ)
                {
                    hnsw_index = std::make_shared<faiss::IndexHNSWSQ>(
                        this->data_dim, quantizer.first, m, this->faiss_metric);
                    SI_LOG_INFO(
                        "Create HNSWSQ Index, m={}, ef_c={}, bit_size={}, "
                        "bits_per_dimension={}",
                        m,
                        ef_c,
                        bit_size,
                        quantizer.second);
                }
                else
                {
                    hnswfast_index = std::make_shared<faiss::IndexHNSWfastSQ>(
                        this->data_dim, quantizer.first, m, this->faiss_metric);
                    SI_LOG_INFO(
                        "Create HNSWfastSQ Index, m={}, ef_c={}, bit_size={}",
                        m,
                        ef_c,
                        bit_size);
                }
                bits_per_dimension = static_cast<int>(quantizer.second);
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            }
            case IndexType::HNSWPQ:
            case IndexType::HNSWfastPQ: {
                size_t M = params.extractParam("M", 32);
                size_t bit_size = params.extractParam("bit_size", 8);
                if (this->index_type == IndexType::HNSWPQ)
                {
                    hnsw_index = std::make_shared<faiss::IndexHNSWPQ>(
                        this->data_dim, M, bit_size, m, this->faiss_metric);
                    SI_LOG_INFO(
                        "Create HNSWPQ Index, PQ_M={} bit_size={} m={}",
                        M,
                        bit_size,
                        m);
                }
                else
                {
                    hnswfast_index = std::make_shared<faiss::IndexHNSWfastPQ>(
                        this->data_dim, M, bit_size, m, this->faiss_metric);
                    SI_LOG_INFO(
                        "Create HNSWfastPQ Index, PQ_M={} bit_size={} m={}",
                        M,
                        bit_size,
                        m);
                }
                bits_per_dimension = static_cast<int>(bit_size);
                num_quantization_blocks = static_cast<int>(M);
                break;
            }
            case IndexType::HNSWFLAT:
                hnsw_index = std::make_shared<faiss::IndexHNSWFlat>(
                    this->data_dim, m, this->faiss_metric);
                SI_LOG_INFO("Create HNSWFlat Index, m={}, ef_c={}", m, ef_c);
                bits_per_dimension = sizeof(float) * this->CHAR_BITS;
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            case IndexType::HNSWfastFLAT:
                hnswfast_index = std::make_shared<faiss::IndexHNSWfastFlat>(
                    this->data_dim, m, this->faiss_metric);
                SI_LOG_INFO(
                    "Create HNSWfastFlat Index, m={}, ef_c={}", m, ef_c);
                bits_per_dimension = sizeof(float) * this->CHAR_BITS;
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            case IndexType::BinaryHNSW:
                binary_hnsw_index = std::make_shared<faiss::IndexBinaryHNSW>(
                    this->data_dim, m, this->faiss_metric);
                SI_LOG_INFO(
                    "Create BinaryHNSW Index, m={}, metric={}",
                    m,
                    enumToString(this->faiss_metric));
                bits_per_dimension = 1;
                num_quantization_blocks = static_cast<int>(this->data_dim);
                break;
            default:
                SI_THROW_MSG(
                    ErrorCode::UNSUPPORTED_PARAMETER,
                    "Unknown HNSW Index Type: " + enumToString(index_type_));
        }
        if (hnsw_index)
        {
            // HNSWFLAT, HNSWSQ, HNSWPQ
            hnsw_index->hnsw.efConstruction = static_cast<int>(ef_c);
            hnsw_index->own_fields = true;
            // downcast to index
            this->setFaissIndex(hnsw_index);
        }
        else if (hnswfast_index)
        {
            // HNSWFLAT, HNSWSQ, HNSWPQ
            hnswfast_index->hnsw.efConstruction = static_cast<int>(ef_c);
            hnswfast_index->own_fields = true;
            // downcast to index
            this->setFaissIndex(hnswfast_index);
        }
        else
        {
            SI_THROW_IF_NOT(binary_hnsw_index, ErrorCode::LOGICAL_ERROR);
            binary_hnsw_index->hnsw.efConstruction = static_cast<int>(ef_c);
            binary_hnsw_index->own_fields = true;
            // downcast to index
            this->setFaissIndex(binary_hnsw_index);
        }

        this->extractAndSetCommonIndexParams(params);
        raiseErrorOnUnknownParams(params);
    }

    SearchParamsPtr
    extractSearchParams(Parameters & params, IDS * filter) override
    {
        using namespace std;

        size_t ef_s = params.extractParam("ef_s", 100);
        bool adaptive_search = params.extractParam("adaptive_search", 0);
        if (adaptive_search && filter)
        {
            float inv_filter_ratio = 1.0f / this->estimateFilterRatio(filter);
            // ef_grow_ratio must grow super-linearly in log space
            float ef_grow_ratio = static_cast<float>(
                pow(1 + pow(log(inv_filter_ratio) / log(m), 1.7), 2.3));
            ef_s = static_cast<size_t>(ef_s * ef_grow_ratio);
        }
        auto * hnsw_params = new faiss::SearchParametersHNSW;
        hnsw_params->efSearch = static_cast<int>(ef_s);
        // hnsw_params->non_selected_skip_dist_ratio =
        //         params.extractParam("non_selected_skip_dist_ratio", -1);

        return SearchParamsPtr(hnsw_params);
    }

    virtual IndexResourceUsage getResourceUsage() const final
    {
        size_t row_bytes = DIV_ROUND_UP(
            bits_per_dimension * num_quantization_blocks, this->CHAR_BITS);
        float theoretical_memory_usage
            = this->maxDataPoints() * (row_bytes + sizeof(uint32_t) * 2 * m)
            + static_cast<size_t>(this->maxDataPoints())
                * static_cast<float>(log(m)) * sizeof(uint32_t);

        float actual_build_memory_usage = theoretical_memory_usage;
        float load_memory_usage;

        if (this->index_type == IndexType::HNSWfastSQ)
        {
            actual_build_memory_usage = theoretical_memory_usage * 1.09f;
            load_memory_usage = theoretical_memory_usage * 1.05;
        }
        else if (this->index_type == IndexType::HNSWfastPQ)
        {
            actual_build_memory_usage = theoretical_memory_usage * 1.22f;
            load_memory_usage = theoretical_memory_usage * 1.14;
        }
        else if (this->index_type == IndexType::HNSWfastFLAT)
        {
            actual_build_memory_usage = theoretical_memory_usage * 1.03f;
            load_memory_usage = theoretical_memory_usage * 1.02;
        }
        else
        {
            /// assume same in other binary condition
            actual_build_memory_usage = theoretical_memory_usage * 1.34f;
            load_memory_usage = theoretical_memory_usage * 1.34;
        }
        auto usage = IndexResourceUsage{
            .memory_usage_bytes = static_cast<size_t>(load_memory_usage),
            .disk_usage_bytes = 0,
            .build_memory_usage_bytes
            = static_cast<size_t>(actual_build_memory_usage),
            .build_disk_usage_bytes = 0,
        };
        SI_LOG_INFO(
            "vector size: {}, level 0 edge {}, other edge {}",
            (this->maxDataPoints() * row_bytes),
            (this->maxDataPoints() * sizeof(uint32_t) * 2 * m),
            (static_cast<size_t>(this->maxDataPoints())
             * static_cast<float>(log(m)) * sizeof(uint32_t)));
        SI_LOG_INFO(
            "HNSW getResourceUsage: {}MB, row_bytes {}, bits_per_dimension {}, "
            "num_quantization_blocks {}",
            usage.memory_usage_bytes >> 20,
            row_bytes,
            bits_per_dimension,
            num_quantization_blocks);
        return usage;
    }

private:
    int bits_per_dimension;
    int num_quantization_blocks;
    size_t m;
    size_t ef_c;
};

/// @brief Faiss flat index to search vectors with bruteforce.
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
class FaissFlatIndex : public FaissIndex<IS, OS, IDS, dataType>
{
public:
    using SearchParamsPtr = std::shared_ptr<faiss::SearchParameters>;

    FaissFlatIndex(
        const std::string & name_,
        IndexType index_type_,
        Metric metric_,
        size_t data_dim_,
        size_t max_points_,
        Parameters params) :
        FaissIndex<IS, OS, IDS, dataType>(
            name_, index_type_, metric_, data_dim_, max_points_)
    {
        switch (this->index_type)
        {
            case IndexType::FLAT:
                this->setFaissIndex(std::make_shared<faiss::IndexFlat>(
                    this->data_dim, this->max_points, this->faiss_metric));
                break;
            case IndexType::BinaryFLAT:
                this->setFaissIndex(std::make_shared<faiss::IndexBinaryFlat>(
                    this->data_dim, this->faiss_metric));
                break;
            default:
                SI_THROW_MSG(
                    ErrorCode::UNSUPPORTED_PARAMETER,
                    "Unknown HNSW Index Type: " + enumToString(index_type_));
        }
        this->extractAndSetCommonIndexParams(params);
        raiseErrorOnUnknownParams(params);
    }

    SearchParamsPtr
    extractSearchParams(Parameters & /*params*/, IDS * /*filter*/) override
    {
        return std::make_shared<faiss::SearchParameters>();
    }

    virtual IndexResourceUsage getResourceUsage() const final
    {
        size_t theoretical_memory_usage = this->getMaxDataBytes();
        float actual_build_memory_usage = theoretical_memory_usage * 1.01f;
        auto load_memory_usage = theoretical_memory_usage;
        auto usage = IndexResourceUsage{
            .memory_usage_bytes = load_memory_usage,
            .disk_usage_bytes = 0,
            .build_memory_usage_bytes
            = static_cast<size_t>(actual_build_memory_usage),
            .build_disk_usage_bytes = 0};
        SI_LOG_INFO(
            "FLAT getResourceUsage: {}MB", usage.memory_usage_bytes >> 20);
        return usage;
    }
};

}
