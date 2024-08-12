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

#include <cmath>
#include <omp.h>
#include <SearchIndex/ScaNNIndex.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#include <scann/base/restrict_allowlist.h>
#include <scann/scann_ops/cc/scann.h>
#include <scann/tree_x_hybrid/tree_x_params.h>
#include <scann/utils/parallel_for.h>
#pragma clang diagnostic pop

namespace Search
{

using research_scann::Deleter;
using research_scann::DenseDataset;

template <typename T>
std::shared_ptr<DenseDataset<T>> build_dataset(
    std::shared_ptr<DenseDataLayer<T>> data_layer, bool memory_mapped = false)
{
    auto storage = research_scann::DenseDataWrapper<T>(
        data_layer->getDataPtr(0),
        data_layer->dataNum() * data_layer->dataDimension(),
        memory_mapped);
    return std::make_shared<DenseDataset<T>>(storage, data_layer->dataNum());
}

template <typename T>
std::unique_ptr<DenseDataset<T>> build_uptr_dataset(
    std::shared_ptr<DenseDataLayer<T>> data_layer, bool memory_mapped = false)
{
    auto storage = research_scann::DenseDataWrapper<T>(
        data_layer->getDataPtr(0),
        data_layer->dataNum() * data_layer->dataDimension(),
        memory_mapped);
    return std::make_unique<DenseDataset<T>>(storage, data_layer->dataNum());
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
ScaNNIndex<IS, OS, IDS, dataType>::ScaNNIndex(
    const std::string & name_,
    Metric metric_,
    size_t data_dim_,
    size_t max_points_,
    Parameters params,
    std::string index_file_prefix_) :
    VectorIndex<IS, OS, IDS, dataType>(
        name_, IndexType::SCANN, metric_, data_dim_, max_points_),
    index_file_prefix(index_file_prefix_)
{
    if (max_points_ < 1000 || max_points_ * data_dim_ < 16000)
    {
        SI_THROW_FMT(
            ErrorCode::BAD_ARGUMENTS,
            "max_points=%lu data_dim=%lu too small for ScaNN, try FLAT Index "
            "instead",
            max_points_,
            data_dim_);
    }
    // data dimension rounded up to multiple of SCANN_DIM_MULTIPLE
    scann_data_dim
        = DIV_ROUND_UP(this->data_dim, SCANN_DIM_MULTIPLE) * SCANN_DIM_MULTIPLE;
    // dimension can't exceed 4096 to avoid overflow errors
    SI_THROW_IF_NOT(scann_data_dim <= 4096, ErrorCode::BAD_ARGUMENTS);
    build_hashed_dataset_by_token
        = params.extractParam("build_hashed_dataset_by_token", true);


    auto version_str
        = params.extractParam("load_index_version", std::string(""));
    if (!version_str.empty())
    {
        SI_LOG_INFO("Loading ScaNN index version {}", version_str);
        auto version_loaded = IndexVersion::fromString(version_str);
        SI_THROW_IF_NOT_FMT(
            version_loaded <= getVersion(),
            ErrorCode::UNSUPPORTED_PARAMETER,
            "Unsupported loaded version %s, current version %s",
            version_loaded.toString().c_str(),
            getVersion().toString().c_str());
    }

    num_children_per_level
        = params.extractParam("num_children_per_level", std::vector<int32_t>());
    if (num_children_per_level.empty())
    {
        // just build a single node when number of data points is small
        if (this->max_points <= 1000)
            num_children_per_level = {1};
        else
            num_children_per_level
                = {static_cast<int32_t>(std::sqrt(this->max_points))};
    }

    min_cluster_size = params.extractParam("min_cluster_size", 50);
    quantization_block_dimension
        = params.extractParam("quantization_block_dimension", 2);
    // training samples are used for clustering and training_sample_size is
    // recommended to be 100 times of num_leaf_nodes
    size_t tss = std::min(
        static_cast<size_t>(getNumLeafNodes() * 75), this->maxDataPoints());
    training_sample_size = params.extractParam("training_sample_size", tss);
    aq_threshold = params.extractParam("aq_threshold", 0.2f);
    // hash_type is either lut16 (4bit) or lut256 (8bit)
    hash_type = params.extractParam("hash_type", std::string("lut16"));
    // turn on residual quantization for IP and Cosine
    residual_quantization = params.extractParam(
        "residual_quantization", this->metric != Metric::L2);
    // random init or kmeans++ for partition center initialization
    // use kmeans++ by default for better accuracy
    partition_random_init = params.extractParam("partition_random_init", false);
    // whether cluster centroids should be quantized
    // turn on by default when we have more than 40 million points
    bool default_qc = this->maxDataPoints() >= 40000000;
    quantize_centroids = params.extractParam("quantize_centroids", default_qc);
    spherical_partition = params.extractParam("spherical_partition", false);
    SI_THROW_IF_NOT(
        HASH_TYPE_PARAMS.contains(hash_type), ErrorCode::UNSUPPORTED_PARAMETER);
    max_top_k = params.extractParam("max_top_k", 2048);
    raiseErrorOnUnknownParams(params);

    // create scann config
    scann_config = std::make_shared<research_scann::ScannConfig>();

    SI_LOG_INFO(
        "Creating ScaNNIndex, "
        "build_hashed_dataset_by_token={} "
        "num_children_per_level={} padded_data_dim={} "
        "training_sample_size={} quantize_centroids={}",
        build_hashed_dataset_by_token,
        vectorToString(num_children_per_level, "_"),
        scann_data_dim,
        training_sample_size,
        quantize_centroids);

    auto u = getResourceUsage();
    SI_LOG_INFO(
        "ScaNNIndex {} estimated resource usage: memory_usage_mb={} "
        "disk_usage_mb={} build_memory_usage_mb={} "
        "build_disk_usage_mb={}",
        this->getName(),
        u.memory_usage_bytes >> 20,
        u.disk_usage_bytes >> 20,
        u.build_memory_usage_bytes >> 20,
        u.build_disk_usage_bytes >> 20);
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
IndexResourceUsage ScaNNIndex<IS, OS, IDS, dataType>::getResourceUsage() const
{
    // compute vector data total bytes
    size_t vec_size = sizeof(T) * scann_data_dim;
    size_t vec_data_bytes = vec_size * this->maxDataPoints();

    // compute hash data size
    auto it = this->HASH_TYPE_PARAMS.find(hash_type);
    size_t row_bits
        = scann_data_dim / quantization_block_dimension * it->second.hash_bits;
    size_t hash_total_bytes
        = this->maxDataPoints() * DIV_ROUND_UP(row_bits, this->CHAR_BITS);
    // used for global residual training
    size_t build_sample_size
        = research_scann::getSampleSize(this->maxDataPoints());
    // kmeans-tree sample bytes
    size_t build_sampling_bytes
        = build_sample_size * scann_data_dim * sizeof(T);
    // chunked vec sample bytes
    build_sampling_bytes += build_sample_size * scann_data_dim * sizeof(T);

    IndexResourceUsage u;
    size_t build_sample_mem_bytes = 0;
    u.memory_usage_bytes = hash_total_bytes * 2 + vec_data_bytes;
    u.build_memory_usage_bytes
        = std::max(u.memory_usage_bytes, build_sampling_bytes);
    SI_LOG_INFO(
        "ScaNNIndex getResourceUsage, build_sample_size={} "
        "build_sampling_bytes={} build_sample_mem_bytes={} "
        "build_memory_usage_bytes={}",
        build_sample_size,
        build_sampling_bytes,
        build_sample_mem_bytes,
        u.build_memory_usage_bytes);
    return u;
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::loadImpl(IndexDataReader<IS> * reader)
{
    this->status = IndexStatus::LOADING;

    SI_LOG_INFO("ScaNNIndex {} starts loading", this->getName());

    // read proto/proto-txt content into message
    auto read_proto_field = [&reader, this](
                                const std::string & name,
                                google::protobuf::Message * message,
                                bool pb_txt = false)
    {
        std::stringstream sstream;
        SI_LOG_INFO("ScaNNIndex starts reading field data {}", name);
        SI_THROW_IF_NOT_FMT(
            reader->readFieldData(index_file_prefix + name, sstream),
            ErrorCode::LOGICAL_ERROR,
            "Read field data %s failed",
            name.c_str());
        bool ok = pb_txt ? ::google::protobuf::TextFormat::ParseFromString(
                      sstream.str(), message)
                         : message->ParseFromIstream(&sstream);
        SI_THROW_IF_NOT_FMT(
            ok,
            ErrorCode::LOGICAL_ERROR,
            "Failed to parse proto from %s",
            name.c_str());
    };

    // read field data into vector, return shape
    auto load_field_vector
        = [&reader, this]<class E>(
              const std::string & name, std::vector<E> & data) -> DataShape
    {
        auto istream
            = reader->getFieldDataInputStream(index_file_prefix + name);
        size_t npts, dim;
        load_bin_from_reader(*istream, data, npts, dim);
        SI_THROW_IF_NOT(npts * dim == data.size(), ErrorCode::LOGICAL_ERROR);
        return {npts, dim};
    };

    // Refer to scann_npy.cc, ScannNumpy::ScannNumpy
    read_proto_field(SCANN_CONFIG_NAME, scann_config.get());
    std::string config_pbtxt = scann_config->DebugString();
    SI_LOG_INFO(
        "{} load with config string:\n{}", this->getName(), config_pbtxt);
    // load setting from config
    auto & partition_config = scann_config->partitioning();
    auto max_num_levels = partition_config.max_num_levels();
    auto & cl = partition_config.num_children_per_level();
    quantize_centroids = partition_config.query_tokenization_type()
        == research_scann::PartitioningConfig_TokenizationType ::
            PartitioningConfig_TokenizationType_FIXED_POINT_INT8;
    num_children_per_level.assign(cl.begin(), cl.end());
    // to be compatible with old config
    if (num_children_per_level.empty())
        num_children_per_level = {partition_config.num_children()};
    SI_THROW_IF_NOT_FMT(
        num_children_per_level.size() == max_num_levels,
        ErrorCode::LOGICAL_ERROR,
        "num_children_per_level size %lu != max_num_levels %d",
        num_children_per_level.size(),
        max_num_levels);

    SI_LOG_INFO(
        "Load from config: num_leaf_nodes={}, max_num_levels={}, "
        "num_children_per_level={}",
        getNumLeafNodes(),
        max_num_levels,
        vectorToString(num_children_per_level, "_"));

    // Refer to scann.cc, ScannInterface::Initialize
    research_scann::SingleMachineFactoryOptions opts;
    research_scann::ScannAssets assets;
    read_proto_field(SCANN_ASSET_NAME, &assets, true);

    auto fp = std::make_shared<research_scann::PreQuantizedFixedPoint>();
    for (const ScannAsset & asset : assets.assets())
    {
        switch (asset.asset_type())
        {
            case ScannAsset::AH_CENTERS:
                opts.ah_codebook = std::make_shared<
                    research_scann::CentersForAllSubspaces>();
                read_proto_field(AH_CODEBOOK_NAME, opts.ah_codebook.get());
                break;
            case ScannAsset::PARTITIONER:
                opts.serialized_partitioner
                    = std::make_shared<research_scann::SerializedPartitioner>();
                read_proto_field(
                    PARTITIONER_NAME, opts.serialized_partitioner.get());
                break;
            case ScannAsset::TOKENIZATION_NPY: {
                std::vector<int32_t> tokenization;
                load_field_vector(TOKENIZATION_NAME, tokenization);
                if (tokenization.empty())
                    break;
                SI_THROW_IF_NOT_MSG(
                    opts.serialized_partitioner != nullptr,
                    ErrorCode::LOGICAL_ERROR,
                    "Non-empty tokenization but no serialized partitioner "
                    "is present.");
                opts.datapoints_by_token = std::make_shared<
                    std::vector<std::vector<research_scann::DatapointIndex>>>(
                    opts.serialized_partitioner->n_tokens());
                for (auto [dp_idx, token] :
                     research_scann::Enumerate(tokenization))
                    opts.datapoints_by_token->at(token).push_back(dp_idx);
                break;
            }
            case ScannAsset::AH_DATASET_NPY: {
                DataShape shape;
                std::string field_name;
                std::shared_ptr<DenseDataset<uint8_t>> * dataset_ptr;

                if (!reader->hasFieldData(
                        index_file_prefix + AH_DATASET_BY_TOKEN_NAME))
                {
                    // load hashed_dataset if hashed_dataset_by_token doesn't exist
                    std::vector<uint8_t> data;
                    shape = load_field_vector(AH_DATASET_NAME, data);
                    field_name = "hashed_dataset";
                    opts.hashed_dataset = make_shared<DenseDataset<uint8_t>>(
                        std::move(data), shape[0]);
                    dataset_ptr = &opts.hashed_dataset;
                }
                else
                {
                    std::function<std::span<uint8_t>(size_t, size_t)> get_data
                        = [&](size_t npts, size_t dim)
                    {
                        createHashedDataLayerIfNeeded(npts, dim);
                        return std::span<uint8_t>(
                            hashed_data_by_token->getDataPtr(0), npts * dim);
                    };
                    auto istream = reader->getFieldDataInputStream(
                        index_file_prefix + AH_DATASET_BY_TOKEN_NAME);
                    load_bin_from_reader(
                        *istream, get_data, shape[0], shape[1]);
                    field_name = "hashed_dataset_by_token";
                    // hashed_dataset_by_token points to hashed_data_by_token internally
                    auto storage = research_scann::DenseDataWrapper<uint8_t>(
                        hashed_data_by_token->getDataPtr(0),
                        shape[0] * shape[1]);
                    opts.hashed_dataset_by_token
                        = std::make_shared<DenseDataset<uint8_t>>(
                            storage, shape[0]);
                    dataset_ptr = &opts.hashed_dataset_by_token;
                }
                SI_LOG_INFO(
                    "Loading {}, shape=({}, {})",
                    field_name,
                    shape[0],
                    shape[1]);
                (*dataset_ptr)->hash_4bit = (hash_type == "lut16");
                size_t num_blocks = DIV_ROUND_UP(
                    this->scann_data_dim, quantization_block_dimension);
                size_t hash_dim = (*dataset_ptr)->hash_4bit
                    ? DIV_ROUND_UP(num_blocks, 2)
                    : num_blocks;

                size_t packed_rows = 0;
                for (auto & t : *opts.datapoints_by_token)
                    packed_rows += (t.size() + 31) & (~31);

                createPackedHashedDataLayerIfNeeded(packed_rows, hash_dim);
                opts.hashed_dataset_packed = build_dataset(hashed_data_packed);

                SI_THROW_IF_NOT_FMT(
                    shape[1] == hash_dim,
                    ErrorCode::LOGICAL_ERROR,
                    "hashed_dataset shape[1] %lu != hash_dim %lu",
                    shape[1],
                    hash_dim);
                break;
            }
            case ScannAsset::DATASET_NPY: {
                SI_THROW_MSG(
                    ErrorCode::LOGICAL_ERROR,
                    "Dataset should be loaded aside from asset!");
                break;
            }
            case ScannAsset::INT8_DATASET_NPY: {
                std::vector<int8_t> data;
                auto shape = load_field_vector(INT8_DATASET_NAME, data);
                fp->fixed_point_dataset
                    = make_shared<research_scann::DenseDataset<int8_t>>(
                        std::move(data), shape[0]);
                break;
            }
            case ScannAsset::INT8_MULTIPLIERS_NPY: {
                std::vector<float> data;
                auto shape = load_field_vector(INT8_MULTIPLIERS_NAME, data);
                fp->multiplier_by_dimension
                    = make_shared<std::vector<float>>(std::move(data));
                break;
            }
            case ScannAsset::INT8_NORMS_NPY: {
                std::vector<float> data;
                auto shape = load_field_vector(DATAPOINT_L2_NORM_NAME, data);
                fp->squared_l2_norm_by_datapoint
                    = make_shared<std::vector<float>>(std::move(data));
                break;
            }
            default:
                break;
        }
    }
    if (fp->fixed_point_dataset != nullptr)
    {
        if (fp->squared_l2_norm_by_datapoint == nullptr)
            fp->squared_l2_norm_by_datapoint
                = std::make_shared<std::vector<float>>();
        opts.pre_quantized_fixed_point = fp;
    }

    createDataLayer();

    data_layer->load(
        reader->getFieldDataInputStream(index_file_prefix + DATASET_NAME).get(),
        opts.hashed_data_size());
    SI_LOG_INFO("ScaNNIndex loaded data_layer, size={}", data_layer->dataNum());

    buildScann(opts);
    // both hashed_data_by_token & hashed_dataset_by_token are active now
    // hahsed_data_by_token is used in serialization

    this->status = IndexStatus::LOADED;
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::serializeImpl(
    IndexDataWriter<OS> * writer)
{
    printMemoryUsage("ScaNN starts serialization");
    // after index build, we create hashed_dataset_by_token for serialization
    // this would incur memory usage when disk_mode == 0|1
    // disk_mode == 2 is therefore recommended to save memory
    if (build_hashed_dataset_by_token)
    {
        // create data layer to store combined hashed data
        createHashedDataLayerIfNeeded();
        auto hashed_dataset_by_token = build_dataset(hashed_data_by_token);
        hashed_dataset_by_token->hash_4bit = (hash_type == "lut16");
        scann->set_hashed_dataset_by_token(hashed_dataset_by_token);
    }
    // Refer to scann.cc ScannInterface::Serialize
    auto status_or = scann->ExtractSingleMachineFactoryOptions();
    SI_THROW_IF_NOT(status_or.ok(), ErrorCode::LOGICAL_ERROR);
    research_scann::SingleMachineFactoryOptions opts
        = std::move(status_or).value();

    research_scann::ScannAssets assets;
    const auto add_asset
        = [&assets](const std::string & fpath, ScannAsset::AssetType type)
    {
        auto * asset = assets.add_assets();
        asset->set_asset_type(type);
        asset->set_asset_path(fpath);
    };

    const auto serialize_asset_pb
        = [&](const std::string & name,
              google::protobuf::Message * message,
              std::optional<ScannAsset::AssetType> asset_type = std::nullopt)
    {
        // skip serialization for empty message
        if (message == nullptr)
            return;
        // manage memory of ostream by AccessibleStringBuf
        AccessibleStringBuf buf;
        std::ostream os(&buf);
        SI_THROW_IF_NOT_FMT(
            message->SerializeToOstream(&os),
            ErrorCode::LOGICAL_ERROR,
            "ScaNN %s serialization failed!",
            name.c_str());
        // write to IndexDataWriter
        auto ostream
            = writer->getFieldDataOutputStream(index_file_prefix + name);
        ostream->write(
            buf.get_internal_buffer(), buf.get_internal_buffer_size());
        // add to asset if it has asset_type
        if (asset_type)
            add_asset(name, asset_type.value());
    };
    serialize_asset_pb(SCANN_CONFIG_NAME, scann_config.get());
    serialize_asset_pb(
        AH_CODEBOOK_NAME, opts.ah_codebook.get(), ScannAsset::AH_CENTERS);
    serialize_asset_pb(
        PARTITIONER_NAME,
        opts.serialized_partitioner.get(),
        ScannAsset::PARTITIONER);

    const auto serialize_asset_data
        = [&]<typename Vec>(
              const std::string & name,
              std::optional<ScannAsset::AssetType> asset_type,
              const Vec & vec,
              std::optional<DataShape> shape = std::nullopt)
    {
        auto ostream
            = writer->getFieldDataOutputStream(index_file_prefix + name);
        auto sizes = shape.value_or(DataShape{vec.size(), 1UL});
        SI_THROW_IF_NOT(
            sizes[0] * sizes[1] == vec.size(), ErrorCode::LOGICAL_ERROR);
        save_bin_with_writer(*ostream, &vec[0], sizes[0], sizes[1]);
        ostream->close();
        if (asset_type)
            add_asset(name, asset_type.value());
    };

    if (opts.datapoints_by_token != nullptr)
    {
        std::vector<int32_t> datapoint_to_token(data_layer->dataNum());
        for (const auto & [token_idx, dps] :
             research_scann::Enumerate(*opts.datapoints_by_token))
            for (auto dp_idx : dps)
                datapoint_to_token[dp_idx] = token_idx;
        serialize_asset_data(
            TOKENIZATION_NAME,
            ScannAsset::TOKENIZATION_NPY,
            datapoint_to_token);
    }

    // prefer to use hashed_dataset_by_token if available
    std::shared_ptr<DenseDataset<uint8_t>> * hashed_dataset_ptr{nullptr};
    std::string field_name;
    if (this->build_hashed_dataset_by_token)
    {
        SI_LOG_INFO("ScaNNIndex::serialize hashed_dataset_by_token");
        SI_THROW_IF_NOT(
            opts.hashed_dataset_by_token != nullptr, ErrorCode::LOGICAL_ERROR);
        hashed_dataset_ptr = &opts.hashed_dataset_by_token;
        field_name = AH_DATASET_BY_TOKEN_NAME;
    }
    else
    {
        SI_LOG_INFO("ScaNNIndex::serialize hashed_dataset");
        SI_THROW_IF_NOT(
            opts.hashed_dataset != nullptr, ErrorCode::LOGICAL_ERROR);
        hashed_dataset_ptr = &opts.hashed_dataset;
        field_name = AH_DATASET_NAME;
    }
    SI_THROW_IF_NOT(hashed_dataset_ptr != nullptr, ErrorCode::LOGICAL_ERROR);
    if (*hashed_dataset_ptr != nullptr)
    {
        auto data = (*hashed_dataset_ptr)->data();
        size_t num = (*hashed_dataset_ptr)->size();
        serialize_asset_data(
            field_name,
            ScannAsset::AH_DATASET_NPY,
            data,
            DataShape{num, data.size() / num});
    }
    if (opts.pre_quantized_fixed_point != nullptr)
    {
        auto fixed_point = opts.pre_quantized_fixed_point;
        auto dataset = fixed_point->fixed_point_dataset;
        if (dataset != nullptr)
        {
            size_t data_size = dataset->data().size();
            serialize_asset_data(
                INT8_DATASET_NAME,
                ScannAsset::INT8_DATASET_NPY,
                dataset->data(),
                DataShape{dataset->size(), data_size / dataset->size()});
        }
        auto multipliers = fixed_point->multiplier_by_dimension;
        if (multipliers != nullptr)
        {
            serialize_asset_data(
                INT8_MULTIPLIERS_NAME,
                ScannAsset::INT8_MULTIPLIERS_NPY,
                *multipliers);
        }
        auto norms = fixed_point->squared_l2_norm_by_datapoint;
        if (norms != nullptr)
        {
            serialize_asset_data(
                DATAPOINT_L2_NORM_NAME, ScannAsset::INT8_NORMS_NPY, *norms);
        }
    }

    // Refer to ScannNumpy::Serialize
    {
        auto assets_ostream = writer->getFieldDataOutputStream(
            index_file_prefix + SCANN_ASSET_NAME);
        auto assets_str = assets.DebugString();
        assets_ostream->write(&assets_str[0], assets_str.size());
        assets_ostream->close();
    }

    {
        // Handle data_layer specially, don't add it to asset
        auto ostream = writer->getFieldDataOutputStream(
            index_file_prefix + DATASET_NAME);
        size_t checksum = data_layer->serialize(ostream.get());
        ostream->close();
        writer->writeFieldChecksum(index_file_prefix + DATASET_NAME, checksum);
    }
    printMemoryUsage("ScaNN finishes serialization");
};


template <typename IS, typename OS, IDSelector IDS, DataType dataType>
typename ScaNNIndex<IS, OS, IDS, dataType>::SearchParams
ScaNNIndex<IS, OS, IDS, dataType>::extractSearchParams(
    int32_t topK,
    Parameters & params,
    bool disk_mode,
    int64_t data_dim,
    int64_t num_data)
{
    // Alpha is a simplified parameter for MyScale database.
    // alpha=1.0: extremely fast but inaccurate, alpha=4.0: fast and accurate
    float alpha = params.extractParam("alpha", 3.0f);
    SI_THROW_IF_NOT_MSG(
        alpha >= 1.0 && alpha <= 4.0,
        ErrorCode::BAD_ARGUMENTS,
        "alpha should be between 1.0 and 4.0");

    // auto-tuned advanced parameters

    // num_reorder grows sublinearly with topK
    uint32_t default_num_reorder =
        static_cast<uint32_t>(
            20 * std::floor(std::pow(topK, 0.65f) * std::sqrt(alpha)));
    // num_reorder grows sublinearly with num_data for large datasets
    if (num_data > 10000000)
        default_num_reorder *= std::pow(num_data / 1e7f, 0.5);
    // reduce num_reorder for high-dimensional datasets
    if (data_dim >= 1024)
        default_num_reorder = default_num_reorder * 768
            / std::min(static_cast<int>(data_dim), 1536);
    // under memory mode, increase num_reorder to increase accuracy
    if (disk_mode == 0)
        default_num_reorder *= 2.5;

    default_num_reorder = std::max(static_cast<uint32_t>(topK), default_num_reorder);
    uint32_t num_reorder
        = params.extractParam("num_reorder", default_num_reorder);
    return {alpha, num_reorder};
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::shared_ptr<SearchResult> ScaNNIndex<IS, OS, IDS, dataType>::searchImpl(
    DataSetPtr & queries,
    int32_t topK,
    Parameters & params,
    bool first_stage_only,
    IDS * filter,
    QueryStats * stats)
{
    using namespace std;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto data_dim = this->dataDimension();
    SI_THROW_IF_NOT(queries->dimension() == data_dim, ErrorCode::BAD_ARGUMENTS);
    auto [alpha, num_reorder]
        = extractSearchParams(topK, params, 0, data_dim, this->numData());
    auto num_leaf_nodes = getNumLeafNodes();

    // default_beta searches for 16% of nodes for 1 level tree
    float default_beta_l = 0.015f;
    if (getMaxNumLevels() == 2)
    {
        // when alpha=3, l_ratio for 10M: 0.06, 20M: 0.037, 50M: 0.019
        default_beta_l = 0.0054f * std::pow(1e7 / this->numData(), 0.7);
    }
    float beta_l = params.extractParam("beta_l", default_beta_l);
    SI_THROW_IF_NOT_FMT(
        beta_l > 0,
        ErrorCode::BAD_ARGUMENTS,
        "beta_l=%f must be larger than 0",
        beta_l);

    // l_search is proportional to num_leaf_nodes, about 16% when alpha=3
    uint32_t default_l_search = static_cast<uint32_t>(
        0.75 * std::floor(1 + num_leaf_nodes * std::exp(alpha * 0.8) * beta_l));
    // exatract l_search param with default value
    uint32_t l_search = params.extractParam("l_search", default_l_search);

    float l_search_ratio = params.extractParam("l_search_ratio", -1.0f);
    bool adaptive_search = params.extractParam("adaptive_search", 1);
    // only use adaptive_search when `l_search_ratio` is not specified in search params
    if (l_search_ratio < 0 && adaptive_search && filter)
    {
        float inv_ratio = 1.0f / this->estimateFilterRatio(filter);
        // when filter ratio is below 0.05, every time it's quartered,
        //   increase l_search by about 1.5x
        // r = log(clip(inv_ratio - 20, 1, 50)) / log(4)
        // l_search := exp(log(l_search) * (0.75 ** r))
        int threshold = data_dim < 1024 ? 200 : (data_dim < 1536 ? 50 : 5);
        // for high dimensional data, we need to increase l_search sooner
        float r
            = log(min(max(inv_ratio - threshold, 1.0f), 1000.0f)) / log(5.0f);
        // for larger dimension, we should decrease gamma and increase l_search_ratio
        // more rapidly as filter_ratio decreases
        float gamma
            = data_dim < 1024 ? 0.85f : (data_dim < 1536 ? 0.78f : 0.7f);
        l_search_ratio
            = exp(log((l_search + 0.01f) / num_leaf_nodes) * pow(gamma, r));
        // hard-code l_search_ratio for extremely large inv_ratio (or extremely low filter ratio)
        int threshold1
            = data_dim < 1024 ? 2000 : (data_dim < 1536 ? 1000 : 500);
        int threshold2
            = data_dim < 1024 ? 5000 : (data_dim < 1536 ? 2000 : 1000);
        if (inv_ratio >= threshold1)
            l_search_ratio = max(l_search_ratio, 0.5f);
        if (inv_ratio >= threshold2)
            l_search_ratio = max(l_search_ratio, 1.0f);
    }
    if (l_search_ratio > 0)
    {
        // calculate l_search from l_search_ratio
        l_search = static_cast<uint32_t>(l_search_ratio * num_leaf_nodes);
        l_search = std::min(l_search, num_leaf_nodes);
    }

    // adjust l_search for large top-k
    auto leaf_nodes = this->getNumLeafNodes();
    if (0.5 * l_search / leaf_nodes * this->max_points < topK) {
        l_search = max(l_search, static_cast<uint32_t>(2 * topK * leaf_nodes / this->max_points));
        l_search = min(l_search, leaf_nodes);
    }

    raiseErrorOnUnknownParams(params);
    SI_LOG_DEBUG(
        "ScaNN::searchImpl with queries of size={} l_search_ratio={} "
        "l_search={} "
        "num_reorder={}",
        queries->numData(),
        l_search_ratio,
        l_search,
        num_reorder);

    if (queries->dimension() != this->scann_data_dim)
        queries = queries->padDataDimension(this->scann_data_dim);

    // get parameters and perform search
    auto scann_params = getScannSearchParametersBatched(
        queries->numData(),
        topK,
        num_reorder,
        l_search,
        true,
        filter);
    std::vector<research_scann::NNResultsVector> res(queries->numData());
    research_scann::DenseDataWrapper<T> queries_data_wrapper(
        const_cast<T *>(queries->getData()),
        queries->numData() * queries->dimension());
    research_scann::DenseDataset<T> queries_dataset(
        queries_data_wrapper, queries->numData());
    research_scann::Status status;
    status = scann->FindNeighborsBatched(
        queries_dataset,
        scann_params,
        research_scann::MakeMutableSpan(res));
    int result_len = topK;
    SI_THROW_IF_NOT_FMT(
        status.ok(),
        ErrorCode::LOGICAL_ERROR,
        "Search error: %s",
        status.error_message().c_str());

    // ScaNN returns -IP for DotProductDistance, so we returns the addictive
    // inverse of the distance for IP and Cosine.
    int sign = this->metric == Metric::L2 ? 1 : -1;
    auto sr = SearchResult::createTopKHolder(queries->numData(), result_len);
    for (size_t i = 0; i < queries->numData(); ++i)
    {
        auto & res_i = res[i];
        SI_THROW_IF_NOT(res_i.size() <= result_len, ErrorCode::LOGICAL_ERROR);

        auto sr_indices_i = sr->getResultIndices(i);
        auto sr_distances_i = sr->getResultDistances(i);
        for (size_t j = 0; j < res_i.size(); ++j)
        {
            sr_indices_i[j] = res_i[j].first;
            sr_distances_i[j] = sign * res_i[j].second;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto time_ms
        = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
              .count();
    return sr;
};

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::setQueryPoolThreads(int num_threads)
{
    return research_scann::QueryThreadPool::setTotalQueryThreads(num_threads);
}


template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::buildImpl(
    IndexSourceDataReader<T> * reader, int num_threads)
{
    SI_LOG_INFO(
        "{} start index building: num_children_per_level={} num_threads={}",
        this->getName(),
        vectorToString(this->num_children_per_level, "_"),
        num_threads);
    printMemoryUsage("ScaNN starts index building");
    createDataLayer();

    // check number of threads used in building
#ifdef MYSCALE_MODE
    SI_THROW_IF_NOT(num_threads > 0, ErrorCode::UNSUPPORTED_PARAMETER);
#else
    // only use openmp in standalone mode
    SI_THROW_IF_NOT(num_threads >= 0, ErrorCode::UNSUPPORTED_PARAMETER);
    if (num_threads == 0)
        num_threads = omp_get_num_procs();
#endif

    size_t block_rows = this->DISK_RW_BLOCK_SIZE / (sizeof(T) * this->data_dim);
    while (!reader->eof())
    {
        auto chunk = reader->readData(block_rows);
        if (chunk->dimension() != this->scann_data_dim)
        {
            chunk = chunk->padDataDimension(this->scann_data_dim);
        }
        if (VectorIndex<IS, OS, IDS, dataType>::metric == Metric::Cosine)
        {
            chunk = chunk->normalize(true);
        }
        data_layer->addData(chunk);
        SIConfiguration::currentThreadCheckAndAbort();
    }
    data_layer->seal();
    SI_LOG_INFO("ScaNN adding data finished");
    // construct config string
    auto config_txt = getScannBuildConfigString();
    SI_LOG_INFO("ScaNN Build Config:\n{}", config_txt.c_str());
    ::google::protobuf::TextFormat::ParseFromString(
        config_txt, scann_config.get());
    // set opts and build scann
    research_scann::SingleMachineFactoryOptions opts;
    opts.parallelization_pool
        = research_scann::StartThreadPool("scann_threadpool", num_threads - 1);

    buildScann(opts);
    printMemoryUsage("ScaNN finishes index building");

    this->status = IndexStatus::SEALED;
};

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::createHashedDataLayerIfNeeded(
    size_t num_rows, size_t hash_dim)
{
    // skip if already created
    if (hashed_data_by_token)
    {
        SI_THROW_IF_NOT(
            hashed_data_by_token->dataNum() >= num_rows,
            ErrorCode::LOGICAL_ERROR);
        SI_THROW_IF_NOT(
            hashed_data_by_token->dataDimension() >= hash_dim,
            ErrorCode::LOGICAL_ERROR);
        return;
    }

    // for each leaf, pad 32 data points at most;
    if (num_rows == 0)
        num_rows = data_layer->dataNum();
    if (hash_dim == 0)
        hash_dim = this->getHashDim();

    hashed_data_by_token = std::make_shared<DenseMemoryDataLayer<uint8_t>>(
        num_rows, hash_dim, /*init_data*/ true);
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::createPackedHashedDataLayerIfNeeded(
    size_t packed_rows, size_t hash_dim)
{
    // skip if already created
    if (hashed_data_packed)
    {
        SI_THROW_IF_NOT(
            hashed_data_packed->dataNum() >= packed_rows,
            ErrorCode::LOGICAL_ERROR);
        SI_THROW_IF_NOT(
            hashed_data_packed->dataDimension() >= hash_dim,
            ErrorCode::LOGICAL_ERROR);
        return;
    }

    // for each leaf, pad 32 data points at most;
    if (packed_rows == 0)
        packed_rows = data_layer->dataNum() + this->getNumLeafNodes() * 32;
    if (hash_dim == 0)
        hash_dim = this->getHashDim();
    hashed_data_packed = std::make_shared<DenseMemoryDataLayer<uint8_t>>(
        packed_rows, hash_dim, /*init_data*/ true);
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
void ScaNNIndex<IS, OS, IDS, dataType>::buildScann(
    research_scann::SingleMachineFactoryOptions & opts)
{
    SI_LOG_INFO(
        "Build index, has_exact_reordering={}",
        scann_config->has_exact_reordering());
    // build dataset
    auto dataset = std::make_shared<research_scann::DenseDataset<T>>(
        research_scann::DenseDataWrapper<T>(data_layer.get()),
        data_layer->dataNum());
    // set normalization tag
    if (dataset && scann_config->has_partitioning()
        && scann_config->partitioning().partitioning_type()
            == research_scann::PartitioningConfig::SPHERICAL)
        dataset->set_normalization_tag(research_scann::UNITL2NORM);
    // build ScaNN index & modify opts in place
    auto status_or
        = SingleMachineFactoryScann<T>(*scann_config, dataset, std::move(opts));
    SI_THROW_IF_NOT_FMT(
        status_or.ok(),
        ErrorCode::LOGICAL_ERROR,
        "ScaNNIndex build failed: %s",
        status_or.status().error_message().c_str());
    // transfer ownership to scann field
    scann.reset(std::move(status_or).value().release());

    const std::string & distance
        = scann_config->distance_measure().distance_measure();
    const absl::flat_hash_set<std::string> negated_distances{
        "DotProductDistance",
        "BinaryDotProductDistance",
        "AbsDotProductDistance",
        "LimitedInnerProductDistance"};
    float result_multiplier
        = negated_distances.find(distance) == negated_distances.end() ? 1 : -1;
};

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::string ScaNNIndex<IS, OS, IDS, dataType>::getScannBuildConfigString()
{
    // Adapted from scann_builder.py
    using Map = typename std::unordered_map<std::string, std::string>;
    auto replace_map = [](const std::string & s, const Map & vars)
    {
        std::string res = s;
        for (auto & p : vars)
        {
            size_t pos = 0;
            while (true)
            {
                pos = res.find(p.first, pos);
                if (pos >= res.size())
                    break;
                res = res.replace(pos, p.first.size(), p.second);
            }
        }
        return res;
    };
    std::string config_txt = "";

    // assuming returning 2048 neighbors at most
    std::string general_config_tpl = R"(
    num_neighbors: {max_top_k}
    distance_measure {distance_measure: {distance_measure}}
)";
    SI_THROW_IF_NOT_FMT(
        METRIC_TO_MEASURE_MAP.contains(this->metric),
        ErrorCode::UNSUPPORTED_PARAMETER,
        "Unsupported metric: %s",
        enumToString(this->metric).c_str());
    auto measure = "\"" + METRIC_TO_MEASURE_MAP.at(this->metric) + "\"";
    Map general_vars = {
        {"{max_top_k}", std::to_string(max_top_k)},
        {"{distance_measure}", measure},
    };
    config_txt += replace_map(general_config_tpl, general_vars);

    std::string partitioning_config_tpl = R"(
    partitioning {
    num_children: {num_children}
    num_children_per_level: {num_children_per_level}
    min_cluster_size: {min_cluster_size}
    max_num_levels: {max_num_levels}
    max_clustering_iterations: 12
    single_machine_center_initialization: {partition_initialization}
    partitioning_distance {
        distance_measure: "SquaredL2Distance"
    }
    query_spilling {
        spilling_type: FIXED_NUMBER_OF_CENTERS
        max_spill_centers: {num_leaf_nodes}
    }
    expected_sample_size: {training_sample_size}
    query_tokenization_distance_override {distance_measure: {distance_measure}}
    partitioning_type: {partition_type}
    query_tokenization_type: {query_tokenization_type}
    }
)";
    int32_t num_children = num_children_per_level.size() == 1
        ? num_children_per_level[0]
        : static_cast<int32_t>(std::sqrt(getNumLeafNodes()));
    Map partition_vars
        = {{"{num_leaf_nodes}", std::to_string(getNumLeafNodes())},
           {"{max_num_levels}", std::to_string(getMaxNumLevels())},
           {"{num_children}", std::to_string(num_children)},
           {"{num_children_per_level}",
            vectorToString(num_children_per_level, ", ", "[", "]")},
           {"{min_cluster_size}", std::to_string(min_cluster_size)},
           {"{partition_initialization}",
            partition_random_init ? "RANDOM_INITIALIZATION"
                                  : "DEFAULT_KMEANS_PLUS_PLUS"},
           {"{training_sample_size}", std::to_string(training_sample_size)},
           {"{distance_measure}", measure},
           {"{partition_type}", spherical_partition ? "SPHERICAL" : "GENERIC"},
           {"{query_tokenization_type}",
            quantize_centroids ? "FIXED_POINT_INT8" : "FLOAT"}};
    config_txt += replace_map(partitioning_config_tpl, partition_vars);

    auto num_blocks = this->scann_data_dim / quantization_block_dimension;
    auto n_dims_remainder = this->scann_data_dim % quantization_block_dimension;
    std::string proj_config_tpl;
    if (n_dims_remainder == 0)
        proj_config_tpl = R"(
    projection_type: CHUNK
    num_blocks: {num_blocks}
    num_dims_per_block: {dimensions_per_block}
)";
    else
        proj_config_tpl = R"(
    projection_type: VARIABLE_CHUNK
    variable_blocks {{
        num_blocks: {num_blocks}
        num_dims_per_block: {dimensions_per_block}
    }}
    variable_blocks {{
        num_blocks: {1}
        num_dims_per_block: {n_dims_remainder}
    }}
)";
    Map proj_vars
        = {{"{num_blocks}", std::to_string(num_blocks)},
           {"{dimensions_per_block}",
            std::to_string(quantization_block_dimension)},
           {"{n_dims_remainder}", std::to_string(n_dims_remainder)}};
    auto proj_config = replace_map(proj_config_tpl, proj_vars);

    std::string hash_config_tpl = R"(
    hash {
    asymmetric_hash {
        lookup_type: {lookup_type}
        use_residual_quantization: {residual_quantization}
        use_global_topn: {global_topn}
        quantization_distance {
        distance_measure: "SquaredL2Distance"
        }
        num_clusters_per_block: {clusters_per_block}
        projection {
        input_dim: {n_dims}
        {proj_config}
        }
        noise_shaping_threshold: {aq_threshold}
        expected_sample_size: {training_sample_size}
        max_clustering_iterations: 10
    }
    }
)";
    auto h = HASH_TYPE_PARAMS.at(hash_type);
    bool global_topn = (hash_type == "lut16") && (num_blocks <= 256)
        && residual_quantization;
    Map hash_vars = {
        {"{lookup_type}", h.lookup_type},
        {"{residual_quantization}", residual_quantization ? "true" : "false"},
        {"{global_topn}", global_topn ? "true" : "false"},
        {"{clusters_per_block}", std::to_string(h.clusters_per_block)},
        {"{n_dims}", std::to_string(this->scann_data_dim)},
        {"{proj_config}", proj_config},
        {"{aq_threshold}", std::to_string(aq_threshold)},
        {"{training_sample_size}", std::to_string(training_sample_size)}};
    auto hash_config = replace_map(hash_config_tpl, hash_vars);
    config_txt += hash_config;

    // during construction, we simply reorder 100 data points
    std::string reorder_config = R"(
    exact_reordering {
    approx_num_neighbors: 100
    fixed_point {
        enabled: False
    }
    }
)";
    config_txt += reorder_config;

    return config_txt;
}

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::vector<research_scann::SearchParameters>
ScaNNIndex<IS, OS, IDS, dataType>::getScannSearchParametersBatched(
    int batch_size,
    int final_nn,
    int pre_reorder_nn,
    int leaves,
    bool set_unspecified,
    IDS * filter) const
{
    // Refer to scann.cc GetSearchParametersBatched()
    std::vector<research_scann::SearchParameters> params(batch_size);
    std::shared_ptr<research_scann::TreeXOptionalParameters> tree_params;
    if (leaves > 0)
    {
        tree_params
            = std::make_shared<research_scann::TreeXOptionalParameters>();
        tree_params->set_num_partitions_to_search_override(leaves);
    }

    std::shared_ptr<research_scann::RestrictAllowlist> whitelist = nullptr;
    if (filter != nullptr)
    {
        static_assert(std::is_same<IDS, DenseBitmap>::value);
        whitelist = std::make_shared<research_scann::RestrictAllowlist>(
            filter, *this->id_list.get());
    }
    for (auto & p : params)
    {
        p.set_pre_reordering_num_neighbors(pre_reorder_nn);
        p.set_post_reordering_num_neighbors(final_nn);
        if (tree_params)
            p.set_searcher_specific_optional_parameters(tree_params);
        p.set_restrict_whitelist(whitelist);
        if (set_unspecified)
            scann->SetUnspecifiedParametersToDefaults(&p);
    }
    return params;
}

// instantiate the template class
template class ScaNNIndex<
    AbstractIStream,
    AbstractOStream,
    DenseBitmap,
    DataType::FloatVector>;

}
