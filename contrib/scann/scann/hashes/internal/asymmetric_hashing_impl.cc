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

#include "scann/hashes/internal/asymmetric_hashing_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>

#include "absl/random/distributions.h"
#include "scann/data_format/datapoint.h"
#include "scann/distance_measures/one_to_many/one_to_many.h"
#include "scann/hashes/internal/asymmetric_hashing_postprocess.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/projection/chunking_projection.h"
#include "scann/utils/common.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/top_n_amortized_constant.h"
#include "scann/utils/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace research_scann {
namespace asymmetric_hashing_internal {

template <typename T>
StatusOr<vector<DenseDataset<double>>> AhImpl<T>::TrainAsymmetricHashing(
    const TypedDataset<T>& dataset, const TrainingOptionsT& opts,
    shared_ptr<ThreadPool> pool) {
  if (dataset.empty()) {
    return InvalidArgumentError("Cannot train AH on an empty dataset.");
  }

  ChunkedDatapoint<float> chunked_vec;

  if (opts.preprocessing_function()) {
    TF_ASSIGN_OR_RETURN(Datapoint<T> preprocessed,
                        opts.preprocessing_function()(dataset[0]));
    SCANN_RETURN_IF_ERROR(
        opts.projector()->ProjectInput(preprocessed.ToPtr(), &chunked_vec));
  } else {
    SCANN_RETURN_IF_ERROR(
        opts.projector()->ProjectInput(dataset[0], &chunked_vec));
  }
  int32_t num_blocks = chunked_vec.size();
  vector<shared_ptr<DenseDataset<float>>> chunked_dataset(num_blocks);
  vector<Deleter> chunked_data_deleter(num_blocks);

  const float sampling_fraction =
      opts.config().has_expected_sample_size()
          ? std::min(1.0,
                     static_cast<double>(opts.config().expected_sample_size()) /
                         static_cast<double>(dataset.size()))
          : opts.config().sampling_fraction();

  MTRandom rng(kDeterministicSeed * (opts.config().sampling_seed() + 1));
  vector<DatapointIndex> sample;
  for (DatapointIndex i = 0; i < dataset.size(); ++i) {
    if (absl::Uniform<double>(rng, 0, 1.0) < sampling_fraction) {
      sample.push_back(i);
    }
  }

  if (sample.size() > opts.config().max_sample_size()) {
    std::shuffle(sample.begin(), sample.end(), rng);
    sample.resize(opts.config().max_sample_size());
    std::sort(sample.begin(), sample.end());
  }

  if (sample.size() < opts.config().num_clusters_per_block()) {
    return InvalidArgumentError(absl::StrCat(
        "Number of clusters per block (",
        opts.config().num_clusters_per_block(),
        ") is greater than asymmetric hashing training data size (",
        sample.size(), ")."));
  }

  ConstSpan<float> weights = opts.weights();
  vector<float> sample_weight_storage;
  if (sample.size() < dataset.size() && !weights.empty()) {
    sample_weight_storage.reserve(sample.size());
    for (DatapointIndex dp_idx : sample) {
      sample_weight_storage.push_back(weights[dp_idx]);
    }
    weights = sample_weight_storage;
  }
  vector<DenseDataset<double>> all_centers(num_blocks);

  bool disk_mode3 = false;
  if (opts.dataset_creator()) {
    Deleter test_deleter;
    disk_mode3 = opts.dataset_creator()(
      "test_chunk", 0, /*disk_level*/ 3, 1, 1, test_deleter) != nullptr;
    if (test_deleter) test_deleter();
  }
  // partition the dimensions into multiple groups to save memory when disk_mode >= 3
  // IO on sample data is doubled, but it should be acceptable
  int32_t num_partitions = disk_mode3 ? 2 : 1;
  int32_t partition_size = (num_blocks + num_partitions - 1) / num_partitions;

  for (int32_t part_begin=0; part_begin<num_blocks; part_begin+=partition_size) {
    int32_t part_end = part_begin + partition_size;
    if (part_end > num_blocks) part_end = num_blocks;

    SI_LOG_INFO("Allocating chunked datasets with sample_size={} sampling_fraction={} "
                "partition_begin={} partition_end={}",
                sample.size(), sampling_fraction, part_begin, part_end);
    bool use_dataset_creator = false;
    for (int32_t i = part_begin; i < part_end; ++i) {
      // try creating dataset on first block
      if (i == 0)
        use_dataset_creator = opts.dataset_creator() != nullptr;
      if (use_dataset_creator) {
        chunked_dataset[i] = opts.dataset_creator()(
          "ResidualChunk", i, /*disk_level*/ 4,
          sample.size(), chunked_vec[i].dimensionality(),
          chunked_data_deleter[i]);
      }
      if (i == 0)
        use_dataset_creator = chunked_dataset[i] != nullptr;
      // if can't use dataset_creator, fallback to memory datasets
      if (!use_dataset_creator) {
        auto ds = std::make_shared<DenseDataset<float>>();
        ds->set_dimensionality(chunked_vec[i].dimensionality());
        ds->Reserve(sample.size());
        chunked_dataset[i] = std::move(ds);
      }
    }

    auto append_chunked_blocks = [&](DatapointIndex dp_idx) {
      for (size_t j = part_begin; j < part_end; ++j) {
        auto cv = chunked_vec[j];
        if (use_dataset_creator)
          std::copy(cv.values(), cv.values() + cv.dimensionality(),
                    chunked_dataset[j]->mutable_data(dp_idx).begin());
        else chunked_dataset[j]->AppendOrDie(cv, "");
      }
    };

    DatapointIndex idx = 0;
    if (opts.preprocessing_function()) {
      for (DatapointIndex i : sample) {
        SCANN_RETURN_IF_ERROR(VerifyAllFinite(dataset[i].values_slice()));
        TF_ASSIGN_OR_RETURN(Datapoint<T> preprocessed,
                            opts.preprocessing_function()(dataset[i]));
        SCANN_RETURN_IF_ERROR(VerifyAllFinite(preprocessed.values()));
        SCANN_RETURN_IF_ERROR(
            opts.projector()->ProjectInput(preprocessed.ToPtr(), &chunked_vec));
        append_chunked_blocks(idx++);
      }
    } else {
      for (DatapointIndex i : sample) {
        SCANN_RETURN_IF_ERROR(VerifyAllFinite(dataset[i].values_slice()));
        SCANN_RETURN_IF_ERROR(
            opts.projector()->ProjectInput(dataset[i], &chunked_vec));
        append_chunked_blocks(idx++);
      }
    }

    auto quantization_distance = opts.quantization_distance();
    GmmUtils::Options gmm_opts;
    gmm_opts.seed = opts.config().clustering_seed();
    gmm_opts.max_iterations = opts.config().max_clustering_iterations();
    gmm_opts.epsilon = opts.config().clustering_convergence_tolerance();
    gmm_opts.parallelization_pool = std::move(pool);
    GmmUtils gmm(quantization_distance, gmm_opts);

    for (size_t i : Seq(part_begin, part_end)) {
      DenseDataset<double> centers;
      vector<vector<DatapointIndex>> subpartitions;
      SCANN_RETURN_IF_ERROR(gmm.ComputeKmeansClustering(
          *chunked_dataset[i], opts.config().num_clusters_per_block(), &centers,
          {.final_partitions = &subpartitions, .weights = weights}));

      for (size_t center_idx : IndicesOf(centers)) {
        SCANN_RETURN_IF_ERROR(
            VerifyAllFinite(centers[center_idx].values_slice()));
        if (!opts.config().use_norm_biasing_correction()) continue;
        TF_ASSIGN_OR_RETURN(
            const double norm_bias_correction,
            ComputeNormBiasCorrection(*chunked_dataset[i], centers[center_idx],
                                      subpartitions[center_idx]));
        SCANN_RET_CHECK(std::isfinite(norm_bias_correction))
            << norm_bias_correction;
        for (double& d : centers.mutable_data(center_idx)) {
          d *= norm_bias_correction;
        }
      }

      chunked_dataset[i]->clear();
      chunked_dataset[i]->ShrinkToFit();
      if (chunked_data_deleter[i])
        chunked_data_deleter[i]();

      vector<uint32_t> centers_permutation(centers.size());
      std::iota(centers_permutation.begin(), centers_permutation.end(), 0U);
      std::sort(centers_permutation.begin(), centers_permutation.end(),
                [&subpartitions](uint32_t a, uint32_t b) {
                  return subpartitions[a].size() > subpartitions[b].size();
                });

      constexpr size_t kAssumedCacheLineSize = 64;
      constexpr size_t kFloatsPerCacheLine =
          kAssumedCacheLineSize / sizeof(float);
      const uint64_t cache_lines_per_row =
          std::max(static_cast<size_t>(1), centers.size() / kFloatsPerCacheLine);
      const size_t num_rotate =
          ((i / 2) % cache_lines_per_row) * kFloatsPerCacheLine;
      std::rotate(centers_permutation.begin(),
                  centers_permutation.begin() + num_rotate,
                  centers_permutation.end());

      if (i & 1) {
        std::reverse(centers_permutation.begin(), centers_permutation.end());
      }

      for (uint32_t j : centers_permutation) {
        all_centers[i].AppendOrDie(centers[j], "");
      }
    }
  } // loop over dim partitions

  return std::move(all_centers);
}

template <typename T>
Status AhImpl<T>::IndexDatapoint(const DatapointPtr<T>& input,
                                 const ChunkingProjection<T>& projection,
                                 const DistanceMeasure& quantization_distance,
                                 ConstSpan<DenseDataset<FloatT>> centers,
                                 MutableSpan<uint8_t> result) {
  DCHECK(!centers.empty());
  ChunkedDatapoint<FloatT> projected;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(input, &projected));

  DCHECK_LE(projected.size(), result.size());

  vector<float> distances(centers[0].size());
  DCHECK_GE(distances.size(), 1);
  DCHECK_LE(distances.size(), 256);
  for (size_t i = 0; i < projected.size(); ++i) {
    DCHECK_EQ(centers[0].size(), centers[i].size());
    size_t closest = 0;
    const DatapointPtr<FloatT> projected_ptr = projected[i];
    const DenseDataset<FloatT>& cur_centers = centers[i];

    const size_t centers_size = cur_centers.size();

    if (projected_ptr.IsSparse()) {
      double closest_distance = numeric_limits<double>::infinity();
      for (size_t j = 0; j < centers_size; ++j) {
        const double distance = quantization_distance.GetDistanceHybrid(
            projected_ptr, cur_centers[j]);
        if (ABSL_PREDICT_FALSE(distance < closest_distance)) {
          closest_distance = distance;
          closest = j;
        }
      }
    } else {
      DCHECK_EQ(distances.size(), cur_centers.size());
      DenseDistanceOneToMany(quantization_distance, projected_ptr, cur_centers,
                             MutableSpan<float>(distances));
      auto min_it = std::min_element(distances.begin(), distances.end());
      closest = min_it - distances.begin();
    }
    result[i] = closest;
  }

  return OkStatus();
}

template <typename T>
Status AhImpl<T>::IndexDatapoint(const DatapointPtr<T>& input,
                                 const ChunkingProjection<T>& projection,
                                 const DistanceMeasure& quantization_distance,
                                 ConstSpan<DenseDataset<FloatT>> centers,
                                 Datapoint<uint8_t>* result) {
  DatapointIndex result_size = centers.size();
  DCHECK_EQ(result_size, projection.num_blocks());
  result->clear();
  result->mutable_values()->resize(result_size, 0);

  return AhImpl<T>::IndexDatapoint(input, projection, quantization_distance,
                                   centers, result->mutable_values_slice());
}

namespace {

template <typename T>
T Square(T x) {
  return x * x;
}

double ComputeParallelCostMultiplier(double t, double squared_l2_norm,
                                     DimensionIndex dims) {
  const double parallel_cost = Square(t) / squared_l2_norm;
  const double perpendicular_cost =
      (1.0 - Square(t) / squared_l2_norm) / (dims - 1.0);
  return parallel_cost / perpendicular_cost;
}

struct SubspaceResidualStats {
  double residual_norm = 0.0;

  double parallel_residual_component = 0.0;
};

template <typename T>
SubspaceResidualStats ComputeResidualStatsForCluster(
    ConstSpan<T> maybe_residual_dptr, ConstSpan<T> original_dptr,
    double inv_norm, ConstSpan<FloatingTypeFor<T>> quantized) {
  DCHECK_EQ(maybe_residual_dptr.size(), quantized.size());
  const size_t dims = maybe_residual_dptr.size();
  SubspaceResidualStats result;
  for (size_t i : Seq(dims)) {
    const double residual_coordinate =
        static_cast<double>(maybe_residual_dptr[i]) -
        static_cast<double>(quantized[i]);
    result.residual_norm += Square(residual_coordinate);
    result.parallel_residual_component +=
        residual_coordinate * original_dptr[i] * inv_norm;
  }
  return result;
}

template <typename T>
StatusOr<vector<std::vector<SubspaceResidualStats>>> ComputeResidualStats(
    DatapointPtr<T> maybe_residual_dptr, DatapointPtr<T> original_dptr,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers,
    const ChunkingProjection<T>& projection) {
  const size_t num_subspaces = centers.size();
  DCHECK_GE(num_subspaces, 1);
  vector<std::vector<SubspaceResidualStats>> result(num_subspaces);
  vector<std::vector<SubspaceResidualStats>> residual_stats(num_subspaces);
  const size_t num_clusters_per_block = centers[0].size();

  using FloatT = FloatingTypeFor<T>;
  ChunkedDatapoint<FloatT> maybe_residual_dptr_chunked;
  ChunkedDatapoint<FloatT> original_dptr_chunked;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(maybe_residual_dptr,
                                                &maybe_residual_dptr_chunked));
  SCANN_RETURN_IF_ERROR(
      projection.ProjectInput(original_dptr, &original_dptr_chunked));
  SCANN_RET_CHECK_EQ(maybe_residual_dptr_chunked.size(), num_subspaces);
  SCANN_RET_CHECK_EQ(original_dptr_chunked.size(), num_subspaces);
  double chunked_norm = 0.0;
  for (size_t subspace_idx : Seq(num_subspaces)) {
    for (FloatT x : original_dptr_chunked[subspace_idx].values_slice()) {
      chunked_norm += Square<double>(x);
    }
  }
  chunked_norm = std::sqrt(chunked_norm);
  double inverse_chunked_norm = 1.0 / chunked_norm;

  for (size_t subspace_idx : Seq(num_subspaces)) {
    auto& cur_subspace_residual_stats = residual_stats[subspace_idx];
    cur_subspace_residual_stats.resize(num_clusters_per_block);
    const DenseDataset<FloatingTypeFor<T>>& cur_subspace_centers =
        centers[subspace_idx];
    for (size_t cluster_idx : Seq(num_clusters_per_block)) {
      ConstSpan<FloatingTypeFor<T>> center =
          cur_subspace_centers[cluster_idx].values_slice();
      ConstSpan<FloatT> maybe_residual_dptr_span =
          maybe_residual_dptr_chunked[subspace_idx].values_slice();
      ConstSpan<FloatT> original_dptr_span =
          original_dptr_chunked[subspace_idx].values_slice();
      cur_subspace_residual_stats[cluster_idx] = ComputeResidualStatsForCluster(
          maybe_residual_dptr_span, original_dptr_span, inverse_chunked_norm,
          center);
    }
  }
  return residual_stats;
}

void InitializeToMinResidualNorm(
    ConstSpan<std::vector<SubspaceResidualStats>> residual_stats,
    MutableSpan<uint8_t> result) {
  DCHECK_EQ(result.size(), residual_stats.size());
  for (size_t subspace_idx : IndicesOf(residual_stats)) {
    auto it = std::min_element(
        residual_stats[subspace_idx].begin(),
        residual_stats[subspace_idx].end(),
        [](const SubspaceResidualStats& a, const SubspaceResidualStats& b) {
          return a.residual_norm < b.residual_norm;
        });
    result[subspace_idx] = it - residual_stats[subspace_idx].begin();
  }
}

double ComputeParallelResidualComponent(
    ConstSpan<uint8_t> quantized,
    ConstSpan<std::vector<SubspaceResidualStats>> residual_stats) {
  double result = 0.0;
  for (size_t subspace_idx : IndicesOf(quantized)) {
    const uint8_t cluster_idx = quantized[subspace_idx];
    result +=
        residual_stats[subspace_idx][cluster_idx].parallel_residual_component;
  }
  return result;
}

struct CoordinateDescentResult {
  uint8_t new_center_idx = 0;
  double cost_delta = 0.0;
  double new_parallel_residual_component = 0.0;
};

CoordinateDescentResult OptimizeSingleSubspace(
    ConstSpan<SubspaceResidualStats> cur_subspace_residual_stats,
    const uint8_t cur_center_idx, const double parallel_residual_component,
    const double parallel_cost_multiplier) {
  CoordinateDescentResult result;
  result.new_center_idx = cur_center_idx;
  result.new_parallel_residual_component = parallel_residual_component;
  const double old_subspace_residual_norm =
      cur_subspace_residual_stats[cur_center_idx].residual_norm;
  const double old_subspace_parallel_component =
      cur_subspace_residual_stats[cur_center_idx].parallel_residual_component;
  for (size_t new_center_idx : IndicesOf(cur_subspace_residual_stats)) {
    if (new_center_idx == cur_center_idx) continue;
    const SubspaceResidualStats& rs =
        cur_subspace_residual_stats[new_center_idx];
    const double new_parallel_residual_component =
        parallel_residual_component - old_subspace_parallel_component +
        rs.parallel_residual_component;
    const double parallel_norm_delta = Square(new_parallel_residual_component) -
                                       Square(parallel_residual_component);
    if (parallel_norm_delta > 0.0) continue;
    const double residual_norm_delta =
        rs.residual_norm - old_subspace_residual_norm;
    const double perpendicular_norm_delta =
        residual_norm_delta - parallel_norm_delta;
    const double cost_delta = parallel_cost_multiplier * parallel_norm_delta +
                              perpendicular_norm_delta;
    if (cost_delta < result.cost_delta) {
      result.new_center_idx = new_center_idx;
      result.cost_delta = cost_delta;
      result.new_parallel_residual_component = new_parallel_residual_component;
    }
  }
  return result;
}

Status ValidateNoiseShapingParams(double threshold, double eta) {
  if (std::isnan(eta) && std::isnan(threshold)) {
    return InvalidArgumentError(
        "Either threshold or eta must be specified for noise-shaped AH "
        "indexing.");
  }
  if (!std::isnan(eta) && !std::isnan(threshold)) {
    return InvalidArgumentError(
        "Threshold and eta may not both be specified for noise-shaped AH "
        "indexing.");
  }
  return OkStatus();
}

}  // namespace

template <typename T>
Status AhImpl<T>::IndexDatapointNoiseShaped(
    const DatapointPtr<T>& maybe_residual_dptr,
    const DatapointPtr<T>& original_dptr,
    const ChunkingProjection<T>& projection,
    ConstSpan<DenseDataset<FloatingTypeFor<T>>> centers, double threshold,
    double eta, MutableSpan<uint8_t> result) {
  SCANN_RET_CHECK_EQ(result.size(), centers.size());
  SCANN_RET_CHECK_EQ(maybe_residual_dptr.dimensionality(),
                     original_dptr.dimensionality());
  TF_ASSIGN_OR_RETURN(auto residual_stats,
                      ComputeResidualStats(maybe_residual_dptr, original_dptr,
                                           centers, projection));

  SCANN_RETURN_IF_ERROR(ValidateNoiseShapingParams(threshold, eta));
  const double parallel_cost_multiplier =
      std::isnan(eta) ? ComputeParallelCostMultiplier(
                            threshold, SquaredL2Norm(original_dptr),
                            original_dptr.dimensionality())
                      : eta;
  InitializeToMinResidualNorm(residual_stats, result);
  double parallel_residual_component =
      ComputeParallelResidualComponent(result, residual_stats);

  vector<uint16_t> subspace_idxs(result.size());
  std::iota(subspace_idxs.begin(), subspace_idxs.end(), 0U);
  vector<double> subspace_residual_norms(result.size());
  for (size_t subspace_idx : IndicesOf(result)) {
    const uint8_t cluster_idx = result[subspace_idx];
    subspace_residual_norms[subspace_idx] =
        residual_stats[subspace_idx][cluster_idx].residual_norm;
  }
  std::vector<uint8_t> result_sorted(result.begin(), result.end());
  ZipSortBranchOptimized(
      std::greater<double>(), subspace_residual_norms.begin(),
      subspace_residual_norms.end(), result_sorted.begin(), result_sorted.end(),
      subspace_idxs.begin(), subspace_idxs.end());

  enum { kMaxRounds = 10 };
  bool cur_round_changes = true;
  for (int round = 0; cur_round_changes && round < kMaxRounds; ++round) {
    cur_round_changes = false;
    for (size_t i : IndicesOf(subspace_idxs)) {
      const size_t subspace_idx = subspace_idxs[i];
      ConstSpan<SubspaceResidualStats> cur_subspace_residual_stats =
          residual_stats[subspace_idx];
      const uint8_t cur_center_idx = result_sorted[i];
      auto subspace_result = OptimizeSingleSubspace(
          cur_subspace_residual_stats, cur_center_idx,
          parallel_residual_component, parallel_cost_multiplier);
      if (subspace_result.new_center_idx != cur_center_idx) {
        parallel_residual_component =
            subspace_result.new_parallel_residual_component;
        result_sorted[i] = subspace_result.new_center_idx;
        cur_round_changes = true;
      }
    }
  }

  double final_residual_norm = 0.0;
  for (size_t i : IndicesOf(result_sorted)) {
    const size_t subspace_idx = subspace_idxs[i];
    const uint8_t center_idx = result_sorted[i];
    result[subspace_idx] = center_idx;
    final_residual_norm +=
        residual_stats[subspace_idx][center_idx].residual_norm;
  }
  return OkStatus();
}

template <typename T>
StatusOr<vector<float>> AhImpl<T>::CreateRawFloatLookupTable(
    const DatapointPtr<T>& query, const ChunkingProjection<T>& projection,
    const DistanceMeasure& lookup_distance,
    ConstSpan<DenseDataset<FloatT>> centers, int32_t num_clusters_per_block) {
  ChunkedDatapoint<FloatT> projected;
  SCANN_RETURN_IF_ERROR(projection.ProjectInput(query, &projected));
  SCANN_RET_CHECK_EQ(centers.size(), projected.size());

  vector<float> result(num_clusters_per_block * projected.size());
  float* result_row_start = result.data();

  for (size_t i = 0; i < centers.size();
       ++i, result_row_start += num_clusters_per_block) {
    const DatapointPtr<FloatT> projected_ptr = projected[i];
    const DenseDataset<FloatT>& cur_centers = centers[i];
    if (projected_ptr.IsSparse()) {
      for (size_t j = 0; j < num_clusters_per_block; ++j) {
        result_row_start[j] = static_cast<float>(
            lookup_distance.GetDistanceHybrid(projected_ptr, cur_centers[j]));
      }
    } else {
      if (lookup_distance.specially_optimized_distance_tag() ==
          DistanceMeasure::LIMITED_INNER_PRODUCT) {
        DenseDistanceOneToMany(
            DotProductDistance(), projected_ptr, cur_centers,
            MutableSpan<float>(result_row_start, num_clusters_per_block));
      } else {
        DenseDistanceOneToMany(
            lookup_distance, projected_ptr, cur_centers,
            MutableSpan<float>(result_row_start, num_clusters_per_block));
      }
    }
  }

  return std::move(result);
}

namespace {
float ComputeMultiplierByQuantile(ConstSpan<float> raw_lookup, float quantile,
                                  int32_t max_integer_value) {
  const size_t k = raw_lookup.size() * (1.0 - quantile) + 1;
  if (k == 1) {
    const float max_abs_lookup_element = std::max(
        std::sqrt(numeric_limits<float>::epsilon()), MaxAbsValue(raw_lookup));
    return max_integer_value / max_abs_lookup_element;
  } else {
    DCHECK_LT(quantile, 1.0f);
    TopNAmortizedConstant<float> tn(k);
    for (auto& elem : raw_lookup) {
      tn.push(std::abs(elem));
    }
    return max_integer_value / tn.exact_bottom();
  }
}

template <typename T, typename Lambda>
inline vector<T> ConvertLookupToFixedPointImpl(ConstSpan<float> raw_lookup,
                                               Lambda convert_to_int_lambda,
                                               float multiplier) {
  constexpr T kBias = FixedPointBias<T>();
  vector<T> result(raw_lookup.size());
  for (size_t i = 0; i < raw_lookup.size(); ++i) {
    result[i] = convert_to_int_lambda(raw_lookup[i] * multiplier) + kBias;
  }
  return result;
}

}  // namespace

template <typename T>
vector<T> ConvertLookupToFixedPoint(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&
        conversion_options,
    float* multiplier) {
  DCHECK_GT(conversion_options.multiplier_quantile(), 0.0f);
  DCHECK_LE(conversion_options.multiplier_quantile(), 1.0f);
  using SignedT = make_signed_t<T>;
  *multiplier = ComputeMultiplierByQuantile(
      raw_lookup, conversion_options.multiplier_quantile(),
      numeric_limits<SignedT>::max());
  constexpr int kRound =
      AsymmetricHasherConfig::FixedPointLUTConversionOptions::ROUND;
  if (conversion_options.multiplier_quantile() == 1.0f) {
    if (conversion_options.float_to_int_conversion_method() == kRound) {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup, [](float f) { return std::lround(f); }, *multiplier);
    } else {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup, [](float f) { return static_cast<SignedT>(f); },
          *multiplier);
    }
  } else {
    auto compress_to_bounds = [](float f) {
      f = std::min<float>(f, numeric_limits<SignedT>::max());
      return std::max<float>(f, numeric_limits<SignedT>::min());
    };
    if (conversion_options.float_to_int_conversion_method() == kRound) {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup,
          [&](float f) {
            return static_cast<SignedT>(std::lround(compress_to_bounds(f)));
          },
          *multiplier);
    } else {
      return ConvertLookupToFixedPointImpl<T>(
          raw_lookup,
          [&](float f) { return static_cast<SignedT>(compress_to_bounds(f)); },
          *multiplier);
    }
  }
}

template vector<uint8_t> ConvertLookupToFixedPoint<uint8_t>(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&,
    float* multiplier);
template vector<uint16_t> ConvertLookupToFixedPoint<uint16_t>(
    ConstSpan<float> raw_lookup,
    const AsymmetricHasherConfig::FixedPointLUTConversionOptions&,
    float* multiplier);

// int16 accumulator can improve search performance by about 10%
bool CanUseInt16Accumulator(ConstSpan<uint8_t> lookup_table,
                            size_t num_blocks) {
  const size_t num_centers_per_block = lookup_table.size() / num_blocks;
  DCHECK_EQ(lookup_table.size() % num_blocks, 0);
  if (num_centers_per_block != 16) return false;
  constexpr uint8_t kBias = FixedPointBias<uint8_t>();

  constexpr size_t kMinMin =
      numeric_limits<int16_t>::min() / numeric_limits<int8_t>::min();
  constexpr size_t kMaxMax =
      numeric_limits<int16_t>::max() / numeric_limits<int8_t>::max();
  constexpr size_t kGuaranteedToWork = (kMaxMax < kMinMin) ? kMaxMax : kMinMin;
  if (num_blocks <= kGuaranteedToWork) return true;

  int32_t sum_of_maxes = 0, sum_of_mins = 0;
  auto block_start = lookup_table.begin();
  for (size_t block = num_blocks; block != 0;
       --block, block_start += num_centers_per_block) {
    int8_t max_val = numeric_limits<int8_t>::min();
    int8_t min_val = numeric_limits<int8_t>::max();
    for (size_t i = 0; i < 16; ++i) {
      const int8_t unbiased = block_start[i] - kBias;
      max_val = std::max<int8_t>(unbiased, max_val);
      min_val = std::min<int8_t>(unbiased, min_val);
    }

    sum_of_mins += static_cast<int32_t>(min_val);
    sum_of_maxes += static_cast<int32_t>(max_val);
  }

  return sum_of_maxes <= numeric_limits<int16_t>::max() &&
         sum_of_mins >= numeric_limits<int16_t>::min();
}

template<bool hash_4bit>
void CreatePackedDataset(
    absl::Span<uint8_t> packed_dataset,
    const DenseDataset<uint8_t>& hashed_database,
    DimensionIndex num_blocks) {
  if (hashed_database.empty()) return;

  // DimensionIndex num_blocks = hashed_database[0].nonzero_entries();
  std::function<uint8_t(const uint8_t*, size_t)> get_value;
  if (hash_4bit) {
    get_value = [](const uint8_t* data, size_t i) {
      // if i % 2 == 0, the value is stored in higher 4 bits, otherwise in lower 4 bits
      return (data[i >> 1] >> ((i & 1) << 2)) & 0xF;
    };
  } else {
    // return the unpacked value stored in one byte
    get_value = [](const uint8_t* data, size_t i) { return data[i]; };
  }

  auto packed_size = num_blocks * ((hashed_database.size() + 31) & (~31)) / 2;
  CHECK_EQ(packed_dataset.size(), packed_size);
  DatapointIndex k = 0;
  for (; k < hashed_database.size() / 32; ++k) {
    size_t start = k * 16 * num_blocks;
    for (size_t j = 0; j < num_blocks; ++j) {
      for (size_t m = 0; m < 16; m++) {
        uint8_t u0 = get_value(hashed_database[k * 32 + m].values(), j);
        uint8_t u1 = get_value(hashed_database[k * 32 + m + 16].values(), j);
        packed_dataset[start + j * 16 + m] = u1 * 16 + u0;
      }
    }
  }

  if (k * 32 < hashed_database.size()) {
    size_t start = k * 16 * num_blocks;
    for (size_t j = 0; j < num_blocks; ++j) {
      for (size_t m = 0; m < 16; m++) {
        DatapointIndex dp_idx = k * 32 + m;
        dp_idx = dp_idx >= hashed_database.size() ? (hashed_database.size() - 1)
                                                  : dp_idx;
        uint8_t u0 = get_value(hashed_database[dp_idx].values(), j);

        dp_idx = k * 32 + m + 16;
        dp_idx = dp_idx >= hashed_database.size() ? (hashed_database.size() - 1)
                                                  : dp_idx;
        uint8_t u1 = get_value(hashed_database[dp_idx].values(), j);
        packed_dataset[start + j * 16 + m] = u1 * 16 + u0;
      }
    }
  }
}

template void CreatePackedDataset<true>(
    absl::Span<uint8_t> packed_dataset,
    const DenseDataset<uint8_t>& hashed_database,
    DimensionIndex num_blocks);
template void CreatePackedDataset<false>(
    absl::Span<uint8_t> packed_dataset,
    const DenseDataset<uint8_t>& hashed_database,
    DimensionIndex num_blocks);

template class UnrestrictedIndexIterator<6, IdentityPostprocessFunctor>;
template class UnrestrictedIndexIterator<6, AddBiasFunctor>;
template class UnrestrictedIndexIterator<6, LimitedInnerFunctor>;
template class PopulateDistancesIterator<6, IdentityPostprocessFunctor>;
template class PopulateDistancesIterator<6, AddBiasFunctor>;
template class PopulateDistancesIterator<6, LimitedInnerFunctor>;

SCANN_INSTANTIATE_TYPED_CLASS(, AhImpl);

}  // namespace asymmetric_hashing_internal
}  // namespace research_scann
