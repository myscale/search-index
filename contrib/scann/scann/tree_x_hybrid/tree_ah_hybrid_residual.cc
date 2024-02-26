// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#include "scann/tree_x_hybrid/tree_ah_hybrid_residual.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/base/search_parameters.h"
#include "scann/base/single_machine_base.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/hashes/asymmetric_hashing2/serialization.h"
#include "scann/hashes/asymmetric_hashing2/training.h"
#include "scann/hashes/asymmetric_hashing2/training_options.h"
#include "scann/oss_wrappers/scann_down_cast.h"
#include "scann/oss_wrappers/scann_status_builder.h"
#include "scann/projection/projection_factory.h"
#include "scann/proto/centers.pb.h"
#include "scann/proto/distance_measure.pb.h"
#include "scann/tree_x_hybrid/internal/batching.h"
#include "scann/tree_x_hybrid/internal/utils.h"
#include "scann/tree_x_hybrid/tree_x_params.h"
#include "scann/utils/common.h"
#include "scann/utils/fast_top_neighbors.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

using asymmetric_hashing2::AsymmetricHashingOptionalParameters;

Status TreeAHHybridResidual::EnableCrowdingImpl(
    ConstSpan<int64_t> datapoint_index_to_crowding_attribute) {
  if (leaf_searchers_.empty()) return OkStatus();
  for (size_t token = 0; token < leaf_searchers_.size(); ++token) {
    ConstSpan<DatapointIndex> cur_leaf_datapoints = datapoints_by_token_[token];
    vector<int64_t> leaf_datapoint_index_to_crowding_attribute(
        cur_leaf_datapoints.size());
    for (size_t i = 0; i < cur_leaf_datapoints.size(); ++i) {
      leaf_datapoint_index_to_crowding_attribute[i] =
          datapoint_index_to_crowding_attribute[cur_leaf_datapoints[i]];
    }
    Status status = leaf_searchers_[token]->EnableCrowding(
        std::move(leaf_datapoint_index_to_crowding_attribute));
    if (!status.ok()) {
      for (size_t i = 0; i <= token; ++i) {
        leaf_searchers_[i]->DisableCrowding();
      }
    }
  }
  return OkStatus();
}

void TreeAHHybridResidual::DisableCrowdingImpl() {
  for (auto& ls : leaf_searchers_) {
    ls->DisableCrowding();
  }
}

Status TreeAHHybridResidual::CheckBuildLeafSearchersPreconditions(
    const AsymmetricHasherConfig& config,
    const KMeansTreeLikePartitioner<float>& partitioner) const {
  if (!leaf_searchers_.empty()) {
    return FailedPreconditionErrorBuilder().LogError()
           << "BuildLeafSearchers must not be called more than once per "
              "instance.";
  }
  if (partitioner.query_tokenization_distance()
          ->specially_optimized_distance_tag() !=
      DistanceMeasure::DOT_PRODUCT) {
    return InvalidArgumentErrorBuilder().LogError()
           << "For TreeAHHybridResidual, partitioner must use "
              "DotProductDistance for query tokenization.";
  }
  if (config.partition_level_confidence_interval_stdevs() != 0.0) {
    LOG(WARNING) << "partition_level_confidence_interval_stdevs has no effect.";
  }
  return OkStatus();
}

namespace {
vector<uint32_t> OrderLeafTokensByCenterNorm(
    const KMeansTreeLikePartitioner<float>& partitioner) {
  vector<float> norms(partitioner.n_tokens());
  std::function<void(const KMeansTreeNode&)> impl =
      [&](const KMeansTreeNode& node) {
        if (node.IsLeaf()) {
          const int32_t leaf_id = node.LeafId();
          DCHECK_LT(leaf_id, norms.size());
          norms[leaf_id] = SquaredL2Norm(node.cur_node_center());
        } else {
          for (const KMeansTreeNode& child : node.Children()) {
            impl(child);
          }
        }
      };

  impl(*partitioner.kmeans_tree()->root());
  vector<uint32_t> perm(norms.size());
  std::iota(perm.begin(), perm.end(), 0U);
  ZipSortBranchOptimized(std::greater<float>(), norms.begin(), norms.end(),
                         perm.begin(), perm.end());
  return perm;
}

template <typename GetResidual>
StatusOr<DenseDataset<float>> ComputeResidualsImpl(
    const DenseDataset<float>& dataset, GetResidual get_residual,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    SingleMachineFactoryOptions* opts=nullptr) {
  const size_t dimensionality = dataset.dimensionality();

  vector<uint32_t> tokens_by_datapoint(dataset.size());
  for (uint32_t token : Seq(datapoints_by_token.size())) {
    for (DatapointIndex dp_idx : datapoints_by_token[token]) {
      DCHECK_EQ(tokens_by_datapoint[dp_idx], 0);
      tokens_by_datapoint[dp_idx] = token;
    }
  }

  // sample size for computing residuals
  // TODO further increasing the sample size helps improve accuracy
  //   but also greatly increases the memory usage
  auto sample_size = getSampleSize(dataset.size());
  LOG(INFO) << "Computing residual sample_size=" << sample_size
            << " total_data=" << dataset.size();

  // sample a subset from 0 .. dataset.size() - 1
  std::vector<DatapointIndex> sample_indices(sample_size);
  {
    // perform efficient sample selection using Reservoir Sampling
    std::mt19937 g(2023);
    std::uniform_int_distribution<uint64_t> dis;
    for (size_t i = 0; i<dataset.size(); ++i) {
      if (i < sample_size) sample_indices[i] = i;
      else {
        uint64_t rdx = dis(g, std::uniform_int_distribution<uint64_t>::param_type(0, i));
        if (rdx < sample_size) sample_indices[rdx] = i;
      }
    }
  }

  LOG(INFO) << "Allocating residuals size=" << sample_size << " dimension=" << dimensionality;
  shared_ptr<DenseDataset<float>> residuals;
  bool preallocated_residuals = true;
  if (opts && opts->float_dataset_creator) {
    Deleter residual_deleter = nullptr;
    residuals = opts->float_dataset_creator(
      "Residuals", /*idx*/ 0, /*disk_level*/ 3, sample_size, dimensionality, residual_deleter);
    if (residuals) residuals->set_data_deleter(residual_deleter);
  }

  if (!residuals) {
    residuals = std::make_shared<DenseDataset<float>>();
    residuals->set_dimensionality(dimensionality);
    residuals->Reserve(sample_size);
    preallocated_residuals = false;
  }

  size_t batch_size = sample_size;
  for (size_t st=0; st < sample_size; st+=batch_size) {
    size_t end = std::min(st+batch_size, sample_size);
    for (size_t dp_idx : Seq(st, end)) {
      auto sample_idx = sample_indices[dp_idx];
      const uint32_t token = tokens_by_datapoint[sample_idx];
      TF_ASSIGN_OR_RETURN(auto residual, get_residual(dataset[sample_idx], token));
      if (preallocated_residuals) {
        // copy the residual to the pre-allocated mmap memory
        std::copy(residual.values(), residual.values() + residual.dimensionality(),
                  residuals->mutable_data(dp_idx).begin());
      }
      else residuals->AppendOrDie(residual, "");
    }
  }
  // TODO be careful about memory leakage & corruption
  return std::move(*residuals);
}

}  // namespace

StatusOr<DenseDataset<float>> TreeAHHybridResidual::ComputeResiduals(
    const DenseDataset<float>& dataset,
    const DenseDataset<float>& kmeans_centers,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    SingleMachineFactoryOptions* opts) {
  DCHECK(!kmeans_centers.empty());
  DCHECK_EQ(kmeans_centers.size(), datapoints_by_token.size());
  DCHECK_EQ(kmeans_centers.dimensionality(), dataset.dimensionality());
  const size_t dimensionality = dataset.dimensionality();
  vector<float> scratch(dimensionality);
  auto get_residual =
      [&](const DatapointPtr<float>& dptr,
          const int32_t token) -> StatusOr<DatapointPtr<float>> {
    ConstSpan<float> datapoint = dptr.values_slice();
    ConstSpan<float> center = kmeans_centers[token].values_slice();
    DCHECK_EQ(center.size(), dimensionality);
    DCHECK_EQ(datapoint.size(), dimensionality);
    for (size_t d : Seq(dimensionality)) {
      scratch[d] = datapoint[d] - center[d];
    }
    return MakeDatapointPtr(scratch);
  };

  return ComputeResidualsImpl(dataset, get_residual, datapoints_by_token, opts);
}

StatusOr<DenseDataset<float>> TreeAHHybridResidual::ComputeResiduals(
    const DenseDataset<float>& dataset,
    const KMeansTreeLikePartitioner<float>* partitioner,
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token,
    bool normalize_residual_by_cluster_stdev,
    SingleMachineFactoryOptions* opts) {
  Datapoint<float> residual;
  auto get_residual =
      [&](const DatapointPtr<float>& dptr,
          const int32_t token) -> StatusOr<DatapointPtr<float>> {
    TF_ASSIGN_OR_RETURN(residual,
                        partitioner->ResidualizeToFloat(
                            dptr, token, normalize_residual_by_cluster_stdev));
    return residual.ToPtr();
  };
  return ComputeResidualsImpl(dataset, get_residual, datapoints_by_token, opts);
}

StatusOr<uint8_t> TreeAHHybridResidual::ComputeGlobalTopNShift(
    ConstSpan<std::vector<DatapointIndex>> datapoints_by_token) {
  size_t largest_partition_size = 0;
  for (const auto& dps_in_partition : datapoints_by_token)
    largest_partition_size =
        std::max(largest_partition_size, dps_in_partition.size());

  uint8_t partition_bits = 1;
  while ((1ull << partition_bits) < datapoints_by_token.size())
    partition_bits++;

  if (partition_bits > 32) {
    return FailedPreconditionError(
        "Too many partitions (%d) to work with global top-N",
        datapoints_by_token.size());
  }
  uint8_t global_topn_shift = 32 - partition_bits;
  if ((1ull << global_topn_shift) < largest_partition_size)
    return FailedPreconditionError(
        "%d partitions and the largest has %d datapoints; too many to be "
        "supported with global top-N.",
        datapoints_by_token.size(), largest_partition_size);
  return global_topn_shift;
}

Status TreeAHHybridResidual::PreprocessQueryIntoParamsUnlocked(
    const DatapointPtr<float>& query, SearchParameters& search_params) const {
  const auto& params =
      search_params
          .searcher_specific_optional_parameters<TreeXOptionalParameters>();
  MaxSpillingConfig config;

  if (params)
    config = params->max_spilling_config();

  vector<KMeansTreeSearchResult> centers_to_search;
  SCANN_RETURN_IF_ERROR(query_tokenizer_->TokensForDatapointWithSpilling(
      query, config, &centers_to_search));

  TF_ASSIGN_OR_RETURN(
      auto shared_lookup_table,
      asymmetric_queryer_->CreateLookupTable(query, lookup_type_tag_));
  search_params.set_unlocked_query_preprocessing_results(
      {make_unique<UnlockedTreeAHHybridResidualPreprocessingResults>(
          std::move(centers_to_search), std::move(shared_lookup_table))});
  return OkStatus();
}

Status TreeAHHybridResidual::BuildLeafSearchers(
    const AsymmetricHasherConfig& config,
    unique_ptr<KMeansTreeLikePartitioner<float>> partitioner,
    shared_ptr<const asymmetric_hashing2::Model<float>> ah_model,
    vector<std::vector<DatapointIndex>> datapoints_by_token,
    const DenseDataset<uint8_t>* hashed_dataset,
    const DenseDataset<uint8_t>* hashed_dataset_by_token,
    const DenseDataset<uint8_t>* hashed_dataset_packed,
    ThreadPool* pool) {
  DCHECK(partitioner);
  SCANN_RETURN_IF_ERROR(
      CheckBuildLeafSearchersPreconditions(config, *partitioner));
  if (config.projection().has_ckmeans_config() &&
      config.projection().ckmeans_config().need_learning()) {
    return FailedPreconditionError(
        "Cannot learn ckmeans when building a TreeAHHybridResidual with "
        "pre-training.");
  }
  TF_ASSIGN_OR_RETURN(shared_ptr<const ChunkingProjection<float>> projector,
                      ChunkingProjectionFactory<float>(config.projection()));
  TF_ASSIGN_OR_RETURN(auto quantization_distance,
                      GetDistanceMeasure(config.quantization_distance()));
  lookup_type_tag_ = config.lookup_type();
  std::function<StatusOr<DatapointPtr<uint8_t>>(DatapointIndex, int32_t,
                                                Datapoint<uint8_t>*)>
      get_hashed_datapoint;
  Datapoint<uint8_t> hashed_dp_storage;
  auto indexer = make_shared<asymmetric_hashing2::Indexer<float>>(
      projector, quantization_distance, ah_model);
  const auto dataset =
      dynamic_cast<const DenseDataset<float>*>(this->dataset());

  const bool normalize_residual_by_cluster_stdev =
      config.use_normalized_residual_quantization();

  if (hashed_dataset) {
    get_hashed_datapoint = [hashed_dataset](DatapointIndex i, int32_t token,
                                            Datapoint<uint8_t>* storage)
        -> StatusOr<DatapointPtr<uint8_t>> { return (*hashed_dataset)[i]; };
  } else {
    if (!this->dataset()) {
      return InvalidArgumentError(
          "At least one of dataset/hashed_dataset must be non-null in "
          "TreeAHHybridResidual::BuildLeafSearchersPreTrained.");
    }
    get_hashed_datapoint =
        [&](DatapointIndex i, int32_t token,
            Datapoint<uint8_t>* storage) -> StatusOr<DatapointPtr<uint8_t>> {
      DCHECK(dataset);
      DatapointPtr<float> original = (*dataset)[i];
      TF_ASSIGN_OR_RETURN(
          Datapoint<float> residual,
          partitioner->ResidualizeToFloat(original, token,
                                          normalize_residual_by_cluster_stdev));
      if (std::isnan(config.noise_shaping_threshold())) {
        SCANN_RETURN_IF_ERROR(indexer->Hash(residual.ToPtr(), storage));
      } else {
        SCANN_RETURN_IF_ERROR(indexer->HashWithNoiseShaping(
            residual.ToPtr(), original, storage,
            {.threshold = config.noise_shaping_threshold()}));
      }
      return storage->ToPtr();
    };
  }

  shared_ptr<DistanceMeasure> lookup_distance =
      std::make_shared<DotProductDistance>();
  asymmetric_queryer_ =
      std::make_shared<asymmetric_hashing2::AsymmetricQueryer<float>>(
          projector, lookup_distance, ah_model);
  leaf_searchers_ = vector<unique_ptr<asymmetric_hashing2::Searcher<float>>>(
      datapoints_by_token.size());

  // calculate cumulative sum of leaf sizes
  std::vector<DatapointIndex> leaf_sum_sizes;
  std::vector<DatapointIndex> leaf_sum_sizes_packed;
  leaf_sum_sizes.push_back(0);
  leaf_sum_sizes_packed.push_back(0);
  for (auto& dp : datapoints_by_token) {
    leaf_sum_sizes.push_back(leaf_sum_sizes.back() + dp.size());
    auto packed_size = (dp.size() + 31) & (~31);
    leaf_sum_sizes_packed.push_back(leaf_sum_sizes_packed.back() + packed_size);
  }

  auto build_leaf_for_token = [&](size_t token) -> Status {
    const absl::Time token_start = absl::Now();
    auto hashed_partition = make_unique<DenseDataset<uint8_t>>();
    if (asymmetric_queryer_->quantization_scheme() ==
        AsymmetricHasherConfig::PRODUCT_AND_PACK) {
      hashed_partition->set_packing_strategy(HashedItem::NIBBLE);
    }
    Datapoint<uint8_t> dp;
    Datapoint<uint8_t> hashed_storage;
    auto& dp_vec = datapoints_by_token[token];

    VLOG(1) << "build_leaf_for_token " << token
            << " hashed_dataset_by_token.4bit="
            << (hashed_dataset_by_token ? hashed_dataset_by_token->hash_4bit : -1)
            << " hashed_dataset_by_token.size="
            << (hashed_dataset_by_token ? static_cast<int64_t>(hashed_dataset_by_token->size()) : -1)
            << " hashed_dataset.4bit="
            << (hashed_dataset ? hashed_dataset->hash_4bit : -1)
            << " hashed_dataset.size="
            << (hashed_dataset ? static_cast<int64_t>(hashed_dataset->size()) : -1);

    if (hashed_dataset_by_token) {
      // generate hashed_partition from hashed_datset_by_token in order
      hashed_partition->hash_4bit = hashed_dataset_by_token->hash_4bit;
      for (DatapointIndex j=leaf_sum_sizes[token]; j<leaf_sum_sizes[token+1]; ++j) {
        auto hashed_dptr = (*hashed_dataset_by_token)[j];
        auto local_status = hashed_partition->Append(hashed_dptr, "");
        SCANN_RETURN_IF_ERROR(local_status);
      }
    }
    else if (hashed_dataset) {
      // get partition from hashed_dataset
      hashed_partition->hash_4bit = hashed_dataset->hash_4bit;
      for (DatapointIndex dp_index : datapoints_by_token[token]) {
        auto hashed_dptr = (*hashed_dataset)[dp_index];
        auto local_status = hashed_partition->Append(hashed_dptr, "");
        SCANN_RETURN_IF_ERROR(local_status);
      }
    }
    else {
      // generate hahsed partition on the fly
      size_t batch_size = dp_vec.size();
      for (size_t st = 0; st < dp_vec.size(); st += batch_size) {
        size_t len = std::min(batch_size, dp_vec.size() - st);
        for (size_t i=st; i<st+len; ++i) {
          auto status_or_hashed_dptr =
              get_hashed_datapoint(dp_vec[i], token, &hashed_storage);
          SCANN_RETURN_IF_ERROR(status_or_hashed_dptr.status());
          auto hashed_dptr = status_or_hashed_dptr.ValueOrDie();
          auto local_status = hashed_partition->Append(hashed_dptr, "");
          SCANN_RETURN_IF_ERROR(local_status);
        }
      }
    }

    asymmetric_hashing2::SearcherOptions<float> opts(asymmetric_queryer_,
                                                     indexer);
    opts.set_asymmetric_lookup_type(lookup_type_tag_);
    opts.set_noise_shaping_threshold(config.noise_shaping_threshold());
    VLOG(1) << "Constructing leaf_searcher token=" << token
            << " hashed_partition.dimensionality=" << hashed_partition->dimensionality()
            << " quantization_scheme=" << asymmetric_queryer_->quantization_scheme()
            << " hash_4bit=" << hashed_partition->hash_4bit
            << " hashed_dataset->hash4bit=" << (hashed_dataset == nullptr ? -1 : hashed_dataset->hash_4bit);

    shared_ptr<DenseDataset<uint8_t>> leaf_hashed_dataset_packed = nullptr;
    if (hashed_dataset_packed) {
      auto leaf_size = leaf_sum_sizes_packed[token+1] - leaf_sum_sizes_packed[token];
      auto storage = DenseDataWrapper<uint8_t>(
        const_cast<uint8_t*>(&hashed_dataset_packed->data(leaf_sum_sizes_packed[token])[0]),
        leaf_size * hashed_dataset_packed->dimensionality());
      leaf_hashed_dataset_packed = std::make_shared<DenseDataset<uint8_t>>(
        std::move(storage), leaf_size);
    }
    leaf_searchers_[token] = make_unique<asymmetric_hashing2::Searcher<float>>(
        nullptr, std::move(hashed_partition),
        leaf_hashed_dataset_packed, std::move(opts),
        default_pre_reordering_num_neighbors(),
        default_pre_reordering_epsilon());
    if (!leaf_searchers_[token]->needs_hashed_dataset()) {
      leaf_searchers_[token]->ReleaseHashedDataset();
    }
    VLOG(1) << "Built leaf searcher " << token + 1 << " of "
            << datapoints_by_token.size()
            << " (size = " << datapoints_by_token[token].size() << " DPs) in "
            << absl::ToDoubleSeconds(absl::Now() - token_start) << " sec.";
    return OkStatus();
  };

  SCANN_RETURN_IF_ERROR(ParallelForWithStatus<1>(IndicesOf(datapoints_by_token),
                                                 pool, build_leaf_for_token));
  for (auto& vec : datapoints_by_token) {
    for (DatapointIndex token : vec) {
      num_datapoints_ = std::max(token + 1, num_datapoints_);
    }
  }

  datapoints_by_token_ = std::move(datapoints_by_token);
  leaf_tokens_by_norm_ = OrderLeafTokensByCenterNorm(*partitioner);
  partitioner->set_tokenization_mode(UntypedPartitioner::QUERY);
  query_tokenizer_ = std::move(partitioner);
  if (this->crowding_enabled()) {
    return EnableCrowdingImpl(this->datapoint_index_to_crowding_attribute());
  }
  return OkStatus();
}

Status TreeAHHybridResidual::FindNeighborsImpl(const DatapointPtr<float>& query,
                                               const SearchParameters& params,
                                               NNResultsVector* result) const {
  auto query_preprocessing_results =
      params.unlocked_query_preprocessing_results<
          UnlockedTreeAHHybridResidualPreprocessingResults>();
  if (query_preprocessing_results) {
    return FindNeighborsInternal1(
        query, params, query_preprocessing_results->centers_to_search(),
        result);
  }

  MaxSpillingConfig config;
  auto tree_x_params =
      params.searcher_specific_optional_parameters<TreeXOptionalParameters>();
  if (tree_x_params) {
    auto config_override = tree_x_params->max_spilling_config();
    if (config_override.valid()) config = config_override;
  }
  vector<KMeansTreeSearchResult> centers_to_search;
  SCANN_RETURN_IF_ERROR(query_tokenizer_->TokensForDatapointWithSpilling(
      query, config, &centers_to_search));
  return FindNeighborsInternal1(query, params, centers_to_search, result);
}

namespace {
using QueryForLeaf = tree_x_internal::QueryForResidualLeaf;

vector<std::vector<QueryForLeaf>> InvertCentersToSearch(
    ConstSpan<vector<KMeansTreeSearchResult>> centers_to_search,
    size_t num_centers) {
  vector<std::vector<QueryForLeaf>> result(num_centers);
  for (DatapointIndex query_index : IndicesOf(centers_to_search)) {
    ConstSpan<KMeansTreeSearchResult> cur_query_centers =
        centers_to_search[query_index];
    for (const auto& center : cur_query_centers) {
      result[center.node->LeafId()].emplace_back(query_index,
                                                 center.distance_to_center);
    }
  }
  return result;
}

template <typename TopN>
inline void AssignResults(TopN* top_n, NNResultsVector* results) {
  top_n->FinishUnsorted(results);
}

}  // namespace

Status TreeAHHybridResidual::FindNeighborsBatchedImpl(
    const TypedDataset<float>& queries, ConstSpan<SearchParameters> params,
    MutableSpan<NNResultsVector> results) const {
  auto t0 = std::chrono::high_resolution_clock::now();
  vector<MaxSpillingConfig> configs(queries.size());
  bool centers_overridden = false;
  for (int i = 0; i < queries.size(); i++) {
    auto tree_x_params =
        params[i]
            .searcher_specific_optional_parameters<TreeXOptionalParameters>();
    if (tree_x_params) {
      auto params_config = tree_x_params->max_spilling_config();
      if (params_config.valid()) {
        configs[i] = params_config;
        centers_overridden = true;
      }
    }
  }

  vector<vector<KMeansTreeSearchResult>> centers_to_search(queries.size());
  if (centers_overridden)
    SCANN_RETURN_IF_ERROR(
        query_tokenizer_->TokensForDatapointWithSpillingBatched(
            queries, configs, MakeMutableSpan(centers_to_search)));
  else
    SCANN_RETURN_IF_ERROR(
        query_tokenizer_->TokensForDatapointWithSpillingBatched(
            queries, vector<MaxSpillingConfig>(), MakeMutableSpan(centers_to_search)));
  VLOG(1) << "centers_to_search[0].size=" << centers_to_search[0].size();
  if (!tree_x_internal::SupportsLowLevelBatching(queries, params) ||
      !leaf_searchers_[0]->lut16_ ||
      leaf_searchers_[0]->opts_.quantization_scheme() ==
          AsymmetricHasherConfig::PRODUCT_AND_BIAS) {
    for (size_t i = 0; i < centers_to_search.size(); ++i) {
      SCANN_RETURN_IF_ERROR(FindNeighborsInternal1(
          queries[i], params[i], centers_to_search[i], &results[i]));
    }
    return OkStatus();
  }
  auto queries_by_leaf =
      InvertCentersToSearch(centers_to_search, query_tokenizer_->n_tokens());
  vector<shared_ptr<const SearcherSpecificOptionalParameters>> lookup_tables(
      queries.size());
  for (size_t i : IndicesOf(queries)) {
    TF_ASSIGN_OR_RETURN(auto lut, asymmetric_queryer_->CreateLookupTable(
                                      queries[i], lookup_type_tag_));
    lookup_tables[i] =
        make_shared<AsymmetricHashingOptionalParameters>(std::move(lut));
  }
  vector<FastTopNeighbors<float>> top_ns;
  vector<FastTopNeighbors<float>::Mutator> mutators(params.size());
  top_ns.reserve(params.size());
  for (const auto& [idx, p] : Enumerate(params)) {
    top_ns.emplace_back(p.pre_reordering_num_neighbors(),
                        p.pre_reordering_epsilon());
    top_ns[idx].AcquireMutator(&mutators[idx]);
  }
  std::mutex mutex;
  std::unordered_map<std::thread::id, vector<NNResultsVector>> thread_leaf_results;
  std::vector<uint32_t> non_empty_leaf_tokens;
  for(auto l: leaf_tokens_by_norm_) {
    // find all the non-empty leaf tokens
    if (!queries_by_leaf[l].empty()) non_empty_leaf_tokens.emplace_back(l);
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  QueryParallelFor<600>(Seq(non_empty_leaf_tokens.size()), [&](size_t ind) {
    auto leaf_token = non_empty_leaf_tokens[ind];
    vector<NNResultsVector> *leaf_results;
    {
      std::lock_guard<std::mutex> lock(mutex);
      auto& r = thread_leaf_results[std::this_thread::get_id()];
      leaf_results = &r;
    }
    ConstSpan<QueryForLeaf> queries_for_cur_leaf = queries_by_leaf[leaf_token];
    if (queries_for_cur_leaf.empty()) return;
    vector<SearchParameters> leaf_params =
        tree_x_internal::CreateParamsSubsetForLeaf<QueryForLeaf>(
            params, mutators, lookup_tables, queries_for_cur_leaf);
    auto get_query = [&queries, &queries_for_cur_leaf](DatapointIndex i) {
      return queries[queries_for_cur_leaf[i].query_index];
    };
    leaf_results->clear();
    leaf_results->resize(leaf_params.size());
    using asymmetric_hashing_internal::IdentityPostprocessFunctor;
    IdentityPostprocessFunctor postprocess;
    auto status =
        leaf_searchers_[leaf_token]
            ->FindNeighborsBatchedInternal<IdentityPostprocessFunctor>(
                get_query, leaf_params, postprocess,
                MakeMutableSpan(*leaf_results));
    SI_THROW_IF_NOT_FMT(
      status.ok(), Search::ErrorCode::LOGICAL_ERROR,
      "Search leaf error %s", status.ToString().c_str());

    // update the results
    std::lock_guard<std::mutex> lock(mutex);
    ConstSpan<DatapointIndex> local_to_global_index =
        datapoints_by_token_[leaf_token];
    auto status_or_partition_stdev =
        query_tokenizer_->ResidualStdevForToken(leaf_token);
    const float partition_stdev = status_or_partition_stdev.ok()
                                      ? status_or_partition_stdev.ValueOrDie()
                                      : 1.0;
    for (size_t j = 0; j < queries_for_cur_leaf.size(); ++j) {
      const DatapointIndex cur_query_index =
          queries_for_cur_leaf[j].query_index;
      tree_x_internal::AddLeafResultsToTopN(
          local_to_global_index, queries_for_cur_leaf[j].distance_to_center,
          partition_stdev, leaf_results->at(j), &mutators[cur_query_index]);
    }
  });
  auto t2 = std::chrono::high_resolution_clock::now();

  for (size_t query_index = 0; query_index < results.size(); ++query_index) {
    mutators[query_index].Release();
    top_ns[query_index].FinishUnsorted(&results[query_index]);
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  auto time_leaf_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  auto time_total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();
  VLOG(1) << "TreeAHHybridResidual::FindNeighborsBatchedImpl"
    << " non_empty_leaf_tokens=" << non_empty_leaf_tokens.size()
    <<  " leaf_time_ms=" << time_leaf_ms << " total_time_ms=" << time_total_ms;
  return OkStatus();
}

Status TreeAHHybridResidual::FindNeighborsInternal1(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<KMeansTreeSearchResult> centers_to_search,
    NNResultsVector* result) const {
  if (params.pre_reordering_crowding_enabled()) {
    return FailedPreconditionError("Crowding is not supported.");
  } else if (enable_global_topn_) {
    FastTopNeighbors<float> top_n(params.pre_reordering_num_neighbors(),
                                  params.pre_reordering_epsilon());
    DCHECK(result);
    SearchParameters leaf_params;
    leaf_params.set_pre_reordering_num_neighbors(
        params.pre_reordering_num_neighbors());
    leaf_params.set_per_crowding_attribute_pre_reordering_num_neighbors(
        params.per_crowding_attribute_pre_reordering_num_neighbors());
    leaf_params.set_pre_reordering_epsilon(top_n.epsilon());
    auto query_preprocessing_results =
        params.unlocked_query_preprocessing_results<
            UnlockedTreeAHHybridResidualPreprocessingResults>();

    shared_ptr<AsymmetricHashingOptionalParameters> leaf_specific_params;
    if (query_preprocessing_results) {
      DCHECK(query_preprocessing_results->lookup_table());
      leaf_specific_params = query_preprocessing_results->lookup_table();
    } else {
      TF_ASSIGN_OR_RETURN(
          auto shared_lookup_table,
          asymmetric_queryer_->CreateLookupTable(query, lookup_type_tag_));
      leaf_specific_params = make_shared<AsymmetricHashingOptionalParameters>(
          std::move(shared_lookup_table));
    }
    leaf_specific_params->SetFastTopNeighbors(&top_n);
    leaf_params.set_searcher_specific_optional_parameters(leaf_specific_params);
    NNResultsVector unused_leaf_results;

    for (size_t i = 0; i < centers_to_search.size(); ++i) {
      const uint32_t token = centers_to_search[i].node->LeafId();
      const float distance_to_center = centers_to_search[i].distance_to_center;
      leaf_specific_params->SetIndexAndBias(token << global_topn_shift_,
                                            distance_to_center);

      TranslateGlobalToLeafLocalWhitelist(params, datapoints_by_token_[token],
                                          &leaf_params);
      SCANN_RETURN_IF_ERROR(
          leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
              query, leaf_params, &unused_leaf_results));
    }

    AssignResults(&top_n, result);

    const uint32_t local_idx_mask = (1u << global_topn_shift_) - 1;
    for (pair<DatapointIndex, float>& idx_dis : *result) {
      uint32_t partition_idx = idx_dis.first >> global_topn_shift_;
      uint32_t local_idx = idx_dis.first & local_idx_mask;
      idx_dis.first = datapoints_by_token_[partition_idx][local_idx];
    }
    return OkStatus();
  } else {
    FastTopNeighbors<float> top_n(params.pre_reordering_num_neighbors(),
                                  params.pre_reordering_epsilon());
    return FindNeighborsInternal2(query, params, centers_to_search,
                                  std::move(top_n), result);
  }
}

template <typename TopN>
Status TreeAHHybridResidual::FindNeighborsInternal2(
    const DatapointPtr<float>& query, const SearchParameters& params,
    ConstSpan<KMeansTreeSearchResult> centers_to_search, TopN top_n,
    NNResultsVector* result) const {
  DCHECK(result);
  SearchParameters leaf_params;
  leaf_params.set_pre_reordering_num_neighbors(
      params.pre_reordering_num_neighbors());
  leaf_params.set_per_crowding_attribute_pre_reordering_num_neighbors(
      params.per_crowding_attribute_pre_reordering_num_neighbors());
  auto query_preprocessing_results =
      params.unlocked_query_preprocessing_results<
          UnlockedTreeAHHybridResidualPreprocessingResults>();
  if (query_preprocessing_results) {
    DCHECK(query_preprocessing_results->lookup_table());
    leaf_params.set_searcher_specific_optional_parameters(
        query_preprocessing_results->lookup_table());
  } else {
    TF_ASSIGN_OR_RETURN(
        auto shared_lookup_table,
        asymmetric_queryer_->CreateLookupTable(query, lookup_type_tag_));
    leaf_params.set_searcher_specific_optional_parameters(
        make_unique<AsymmetricHashingOptionalParameters>(
            std::move(shared_lookup_table)));
  }
  typename TopN::Mutator mutator;
  top_n.AcquireMutator(&mutator);
  for (size_t i = 0; i < centers_to_search.size(); ++i) {
    const int32_t token = centers_to_search[i].node->LeafId();
    NNResultsVector leaf_results;
    const float distance_to_center = centers_to_search[i].distance_to_center;
    leaf_params.set_pre_reordering_epsilon(mutator.epsilon() -
                                           distance_to_center);
    TranslateGlobalToLeafLocalWhitelist(params, datapoints_by_token_[token],
                                        &leaf_params);
    SCANN_RETURN_IF_ERROR(
        leaf_searchers_[token]->FindNeighborsNoSortNoExactReorder(
            query, leaf_params, &leaf_results));
    float cluster_stdev_adjustment = centers_to_search[i].residual_stdev;
    tree_x_internal::AddLeafResultsToTopN(
        datapoints_by_token_[token], distance_to_center,
        cluster_stdev_adjustment, leaf_results, &mutator);
  }
  mutator.Release();

  AssignResults(&top_n, result);
  return OkStatus();
}

StatusOr<pair<int32_t, DatapointPtr<float>>>
TreeAHHybridResidual::TokenizeAndMaybeResidualize(
    const DatapointPtr<float>& dptr, Datapoint<float>* residual_storage) {
  KMeansTreeSearchResult token_storage;
  SCANN_RETURN_IF_ERROR(
      database_tokenizer_->TokenForDatapoint(dptr, &token_storage));
  residual_storage->clear();
  auto& vals = *residual_storage->mutable_values();
  vals.resize(dptr.values_slice().size());
  auto center = token_storage.node->cur_node_center();
  for (size_t i : IndicesOf(vals)) {
    vals[i] = dptr.values()[i] - center.values()[i];
  }
  return std::make_pair(token_storage.node->LeafId(),
                        residual_storage->ToPtr());
}

StatusOr<vector<pair<int32_t, DatapointPtr<float>>>>
TreeAHHybridResidual::TokenizeAndMaybeResidualize(
    const TypedDataset<float>& dps,
    MutableSpan<Datapoint<float>*> residual_storage) {
  SCANN_RET_CHECK_EQ(dps.size(), residual_storage.size());
  vector<KMeansTreeSearchResult> token_storage(dps.size());
  SCANN_RETURN_IF_ERROR(
      database_tokenizer_->TokenForDatapointBatched(dps, &token_storage));
  vector<pair<int32_t, DatapointPtr<float>>> result(dps.size());
  for (size_t dp_idx : IndicesOf(residual_storage)) {
    DatapointPtr<float> dptr = dps[dp_idx];
    std::vector<float>& vals = *residual_storage[dp_idx]->mutable_values();
    vals.resize(dptr.values_slice().size());
    auto center = token_storage[dp_idx].node->cur_node_center();
    for (size_t dim_idx : IndicesOf(vals)) {
      vals[dim_idx] = dptr.values()[dim_idx] - center.values()[dim_idx];
    }
    result[dp_idx] = {token_storage[dp_idx].node->LeafId(),
                      residual_storage[dp_idx]->ToPtr()};
  }
  return result;
}

StatusOr<SingleMachineFactoryOptions>
TreeAHHybridResidual::ExtractSingleMachineFactoryOptions() {
  TF_ASSIGN_OR_RETURN(const int dataset_size,
                      UntypedSingleMachineSearcherBase::DatasetSize());
  TF_ASSIGN_OR_RETURN(
      SingleMachineFactoryOptions leaf_opts,
      MergeAHLeafOptions(leaf_searchers_, datapoints_by_token_,
                         hashed_dataset_by_token(), dataset_size));
  TF_ASSIGN_OR_RETURN(
      auto opts,
      SingleMachineSearcherBase<float>::ExtractSingleMachineFactoryOptions());
  opts.datapoints_by_token =
      std::make_shared<vector<std::vector<DatapointIndex>>>(
          datapoints_by_token_);
  opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
  query_tokenizer_->CopyToProto(opts.serialized_partitioner.get());

  if (leaf_opts.ah_codebook != nullptr) {
    opts.ah_codebook = leaf_opts.ah_codebook;
    opts.hashed_dataset = leaf_opts.hashed_dataset;
    opts.hashed_dataset_by_token = leaf_opts.hashed_dataset_by_token;
  }
  return opts;
}

void TreeAHHybridResidual::AttemptEnableGlobalTopN() {
  if (datapoints_by_token_.empty()) {
    LOG(ERROR) << "datapoints_by_token_ is empty. EnableGlobalTopN() should be "
                  "called after all leaves are trained and initialized.";
    return;
  }
  StatusOr<uint8_t> status_or_shift =
      ComputeGlobalTopNShift(datapoints_by_token_);
  if (!status_or_shift.ok()) {
    LOG(ERROR) << "Cannot enable global top-N: " << status_or_shift.status();
    return;
  }
  global_topn_shift_ = status_or_shift.ValueOrDie();
  enable_global_topn_ = true;
}

}  // namespace research_scann
