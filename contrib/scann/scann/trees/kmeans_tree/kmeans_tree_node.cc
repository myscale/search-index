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



#include "scann/trees/kmeans_tree/kmeans_tree_node.h"

#include <math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/one_to_one/l2_distance.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/gmm_utils.h"
#include "scann/utils/parallel_for.h"
#include "scann/utils/scalar_quantization_helpers.h"
#include "scann/utils/types.h"
#include "scann/utils/util_functions.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace research_scann {

KMeansTreeNode::KMeansTreeNode() {}

void KMeansTreeNode::Reset() {
  leaf_id_ = -1;
  learned_spilling_threshold_ = numeric_limits<double>::quiet_NaN();
  indices_.clear();
  children_.clear();
  residual_stdevs_.clear();
}

void KMeansTreeNode::UnionIndices(vector<DatapointIndex>* result) const {
  CHECK(result);
  absl::flat_hash_set<DatapointIndex> union_hash;
  UnionIndicesImpl(&union_hash);
  result->clear();
  for (DatapointIndex elem : union_hash) {
    result->push_back(elem);
  }
}

namespace {

template <typename T>
Datapoint<float> ToDatapoint(google::protobuf::RepeatedField<T> values) {
  Datapoint<float> dp;
  dp.mutable_values()->reserve(values.size());
  for (const auto& elem : values) {
    dp.mutable_values()->push_back(elem);
  }
  return dp;
}

}  // namespace

void KMeansTreeNode::BuildFromProto(const SerializedKMeansTree::Node& proto, int32_t node_level) {
  this->node_level_ = node_level;
  VLOG(1) << "KMeansTreeNode::BuildFromProto() node_level=" << node_level;
  float_centers_.clear();
  Datapoint<float> dp;
  for (size_t i = 0; i < proto.centers_size(); ++i) {
    if (!proto.centers(i).float_dimension().empty()) {
      dp = ToDatapoint(proto.centers(i).float_dimension());
    } else {
      dp = ToDatapoint(proto.centers(i).dimension());
    }

    if (i == 0) {
      float_centers_.set_dimensionality(dp.dimensionality());
      float_centers_.Reserve(proto.centers_size());
    }

    float_centers_.AppendOrDie(dp.ToPtr(), "");
  }

  MaybeInitializeThreadSharding();
  learned_spilling_threshold_ = proto.learned_spilling_threshold();
  leaf_id_ = proto.leaf_id();

  indices_.clear();
  children_.clear();
  residual_stdevs_.clear();
  residual_stdevs_.insert(residual_stdevs_.begin(),
                          proto.residual_stdevs().begin(),
                          proto.residual_stdevs().end());
  if (proto.children_size() == 0) {
    indices_.insert(indices_.begin(), proto.indices().begin(),
                    proto.indices().end());
  } else {
    children_ = vector<KMeansTreeNode>(proto.children_size());
    for (size_t i = 0; i < proto.children_size(); ++i) {
      children_[i].BuildFromProto(proto.children(i), node_level+1);
    }
  }
}

namespace kmeans_tree_internal {

Status PostprocessDistancesForSpilling(
    ConstSpan<float> distances, QuerySpillingConfig::SpillingType spilling_type,
    double spilling_threshold, int32_t max_centers,
    std::vector<pair<DatapointIndex, float>>* child_centers) {
  float epsilon = std::numeric_limits<float>::infinity();
  if (spilling_type != QuerySpillingConfig::NO_SPILLING &&
      spilling_type != QuerySpillingConfig::FIXED_NUMBER_OF_CENTERS) {
    const size_t nearest_center_index =
        std::distance(distances.begin(),
                      std::min_element(distances.begin(), distances.end()));
    const float nearest_center_distance = distances[nearest_center_index];

    using cast_ops::DoubleToFloat;

    float spill_thresh = std::nextafter(DoubleToFloat(spilling_threshold),
                                        std::numeric_limits<float>::infinity());
    TF_ASSIGN_OR_RETURN(
        float max_dist_to_consider,
        ComputeThreshold(nearest_center_distance, spill_thresh, spilling_type));
    epsilon = std::nextafter(max_dist_to_consider,
                             std::numeric_limits<float>::infinity());
  }
  const int32_t max_results =
      (spilling_type == QuerySpillingConfig::NO_SPILLING) ? 1 : max_centers;
  FastTopNeighbors<float> top_n(max_results, epsilon);
  top_n.PushBlock(distances, 0);
  top_n.FinishUnsorted(child_centers);
  return OkStatus();
}

}  // namespace kmeans_tree_internal

Status KMeansTreeNode::Train(const Dataset& training_data,
                             vector<DatapointIndex> subset,
                             const DistanceMeasure& training_distance,
                             const KMeansTreeConfig& config, int32_t current_level,
                             KMeansTreeTrainingOptions* opts) {
  this->node_level_ = current_level;
  indices_ = std::move(subset);
  if (indices_.size() <= opts->max_leaf_size) {
    return OkStatus();
  }

  if (opts->max_num_levels <= current_level) {
    return OkStatus();
  }

  VLOG(1) << "KMeansTreeNode::Train k_per_level=" << config.k_per_level
          << " current_level=" << current_level
          << " data_size=" << training_data.size();

  int k_per_level = config.k_per_level;
  auto& level_nodes = config.num_children_per_level.empty() ?
    SIConfiguration::SCANN_LEVEL_NUM_LEAF_NODES : config.num_children_per_level;
  if (current_level < level_nodes.size() && level_nodes[current_level] > 0) {
    k_per_level = level_nodes[current_level];
    LOG(INFO) << "Override k_per_level=" << k_per_level << " current_level=" << current_level
              << " subset_size=" << indices_.size();
  }

  GmmUtils::Options gmm_opts;
  gmm_opts.max_iterations = opts->max_iterations;
  gmm_opts.epsilon = opts->convergence_epsilon;
  gmm_opts.max_iteration_duration = opts->max_iteration_duration;
  gmm_opts.seed = opts->seed + kDeterministicSeed;
  gmm_opts.min_cluster_size = opts->min_cluster_size;
  gmm_opts.parallelization_pool = opts->training_parallelization_pool;
  gmm_opts.partition_assignment_type = opts->balancing_type;
  gmm_opts.center_reassignment_type = opts->reassignment_type;
  gmm_opts.center_initialization_type = opts->center_initialization_type;
  GmmUtils gmm(MakeDummyShared(&training_distance), gmm_opts);

  vector<vector<DatapointIndex>> subpartitions;
  DenseDataset<double> centers;

  // subsample training set for faster kmeans (only for root data)
  vector<DatapointIndex> training_subset;
  if (current_level == 0 && indices_.size() >= k_per_level * KMEANS_TREE_NODE_SAMPLE_SIZE * 2) {
    size_t inv_ratio = indices_.size() / (k_per_level * KMEANS_TREE_NODE_SAMPLE_SIZE);
    for (size_t i = 0; i < indices_.size(); ++i) {
      if ((i / 16) % inv_ratio == 0) {
        training_subset.push_back(indices_[i]);
      }
    }
    LOG(INFO) << "Tree training_subset inv_ratio=" << inv_ratio
              << " indices_size=" << indices_.size()
              << " subset_size=" << training_subset.size();
  }

  SCANN_RETURN_IF_ERROR(gmm.ComputeKmeansClustering(
      training_data, k_per_level, &centers,
      {.subset = training_subset.empty() ? indices_ : training_subset,
       .final_partitions = &subpartitions,
       .spherical = opts->partitioning_type == PartitioningConfig::SPHERICAL}));

  if (!training_subset.empty()) {
    // do one final pass to compute the centroids for each point
    gmm.ComputeKmeansClustering(
          training_data, k_per_level, &centers,
          {.subset = indices_,
          .final_partitions = &subpartitions,
          .spherical = opts->partitioning_type == PartitioningConfig::SPHERICAL,
          .one_pass_assignment_only = true});
  }

  DatabaseSpillingConfig::SpillingType spilling_type =
      opts->learned_spilling_type;
  if (spilling_type != DatabaseSpillingConfig::NO_SPILLING &&
      opts->per_node_spilling_factor > 1.0) {
    TF_ASSIGN_OR_RETURN(
        learned_spilling_threshold_,
        gmm.ComputeSpillingThreshold(
            training_data, indices_, centers, opts->learned_spilling_type,
            opts->per_node_spilling_factor, opts->max_spill_centers));
  }

  if (spilling_type != DatabaseSpillingConfig::NO_SPILLING &&
      opts->per_node_spilling_factor > 1.0) {
    vector<vector<DatapointIndex>> spilled(centers.size());
    for (DatapointIndex i : indices_) {
      Datapoint<double> double_dp;
      training_data.GetDatapoint(i, &double_dp);
      vector<pair<DatapointIndex, float>> spill_centers;

      {
        std::vector<float> tmp_dists(centers.size());
        kmeans_tree_internal::GetAllDistancesFloatingPointNoThreadSharding<
            double, float>(training_distance, double_dp.ToPtr(), centers,
                           MakeMutableSpan(tmp_dists));

        SCANN_RETURN_IF_ERROR(
            kmeans_tree_internal::PostprocessDistancesForSpilling(
                tmp_dists,
                static_cast<QuerySpillingConfig::SpillingType>(spilling_type),
                learned_spilling_threshold_, opts->max_spill_centers,
                &spill_centers));
      }

      for (const auto& center_index : spill_centers) {
        spilled[center_index.first].push_back(i);
      }
    }

    const size_t max_subpartition_size =
        static_cast<size_t>(floor(0.99 * indices_.size()));
    for (const auto& subpartition : spilled) {
      if (subpartition.size() >= max_subpartition_size) {
        LOG(INFO) << "KILL SPILL " << subpartition.size();
        learned_spilling_threshold_ = NAN;
        spilling_type = DatabaseSpillingConfig::NO_SPILLING;
        break;
      }
    }

    if (!std::isnan(learned_spilling_threshold_)) {
      subpartitions.swap(spilled);
    }
  }

  if (opts->compute_residual_stdev) {
    residual_stdevs_.resize(centers.size());
    ParallelFor<1>(Seq(centers.size()),
                   opts->training_parallelization_pool.get(), [&](size_t i) {
                     double sq_residual_sum = 0.0;
                     uint32_t count = subpartitions[i].size();
                     DatapointPtr<double> center = centers.at(i);
                     for (auto j : subpartitions[i]) {
                       Datapoint<double> double_dp;
                       training_data.GetDatapoint(j, &double_dp);
                       sq_residual_sum +=
                           SquaredL2DistanceBetween(double_dp.ToPtr(), center);
                     }

                     residual_stdevs_[i] = std::max(
                         count == 0
                             ? 1.0
                             : count == 1
                                   ? std::sqrt(sq_residual_sum)
                                   : std::sqrt(sq_residual_sum / (count - 1)),
                         opts->residual_stdev_min_value);
                   });
  }

  FreeBackingStorage(&indices_);
  children_ = vector<KMeansTreeNode>(centers.size());
  VLOG(1) << "Creating KMeansTreeNode size=" << children_.size();
  for (size_t i = 0; i < children_.size(); ++i) {
    children_[i].Reset();
    Status status;
    auto *float_training_data = dynamic_cast<const DenseDataset<float>*>(&training_data);
    if (current_level == 0 && float_training_data
        && float_training_data->isMemoryMapped()) {
      VLOG(1) << "Tree::train copy training_data for children=" << i;
      // copy from mmap dataset
      DenseDataset<float> child_data;
      vector<DatapointIndex> child_indices;
      child_data.set_dimensionality(training_data.dimensionality());
      child_data.Reserve(subpartitions[i].size());
      uint32_t ind = 0;
      for (auto idx : subpartitions[i]) {
        child_data.AppendOrDie((*float_training_data)[idx], "");
        child_indices.push_back(ind++);
      }
      status = children_[i].Train(
        child_data, std::move(child_indices), training_distance,
        config, current_level + 1, opts);
    }
    else {
      // train using memory dataset
      status = children_[i].Train(
        training_data, std::move(subpartitions[i]), training_distance,
        config, current_level + 1, opts);
    }
    if (!status.ok()) return status;
  }

  centers.ConvertType(&float_centers_);
  MaybeInitializeThreadSharding();
  return OkStatus();
}

void KMeansTreeNode::CreateFixedPointCenters() {
  if (!fixed_point_centers_.empty()) return;

  center_squared_l2_norms_.resize(float_centers_.size());
  for (auto [i, norm] : Enumerate(center_squared_l2_norms_))
    norm = SquaredL2Norm(float_centers_[i]);
  ScalarQuantizationResults results =
      ScalarQuantizeFloatDataset(float_centers_, 1.0, NAN);
  inv_int8_multipliers_ = std::move(results.inverse_multiplier_by_dimension);
  fixed_point_centers_ = std::move(results.quantized_dataset);

  for (KMeansTreeNode& child : children_) {
    child.CreateFixedPointCenters();
  }
}

Status KMeansTreeNode::CheckDimensionality(DimensionIndex query_dims) const {
  if (float_centers_.empty()) {
    return OkStatus();
  } else if (float_centers_.dimensionality() == query_dims) {
    return OkStatus();
  } else {
    const std::string error_msg =
        StrFormat("Incorrect query dimensionality.  Expected %u, got %u.\n",
                  static_cast<uint64_t>(float_centers_.dimensionality()),
                  static_cast<uint64_t>(query_dims));
    return FailedPreconditionError(error_msg);
  }
}

int32_t KMeansTreeNode::NumberLeaves(int32_t m) {
  if (IsLeaf()) {
    leaf_id_ = m;
    return m + 1;
  } else {
    leaf_id_ = -1;
    for (KMeansTreeNode& child : children_) {
      m = child.NumberLeaves(m);
    }
  }

  return m;
}

void KMeansTreeNode::PopulateCurNodeCenters() {
  for (size_t i = 0; i < children_.size(); ++i) {
    children_[i].cur_node_center_ = Centers()[i];
    children_[i].PopulateCurNodeCenters();
  }
}

void KMeansTreeNode::CopyToProto(SerializedKMeansTree::Node* proto,
                                 bool with_indices) const {
  CHECK(proto != nullptr);
  for (DatapointIndex i = 0; i < float_centers_.size(); ++i) {
    const DatapointPtr<float> center = float_centers_[i];
    DCHECK(center.IsDense());
    auto center_proto = proto->add_centers();
    for (const float& elem : center.values_slice()) {
      center_proto->add_dimension(elem);
    }
  }

  proto->set_leaf_id(leaf_id_);
  proto->set_learned_spilling_threshold(learned_spilling_threshold_);

  for (const double& residual_stdev : residual_stdevs_) {
    proto->add_residual_stdevs(residual_stdev);
  }

  if (IsLeaf() && with_indices) {
    for (const auto& index : indices_) {
      proto->add_indices(index);
    }
  } else {
    for (const auto& child : children_) {
      auto child_proto = proto->add_children();
      child.CopyToProto(child_proto, with_indices);
    }
  }
}

void KMeansTreeNode::UnionIndicesImpl(
    absl::flat_hash_set<DatapointIndex>* union_hash) const {
  CHECK(union_hash);
  if (IsLeaf()) {
    for (auto index : indices_) {
      union_hash->insert(index);
    }
  } else {
    for (const auto& child : children_) {
      child.UnionIndicesImpl(union_hash);
    }
  }
}

void KMeansTreeNode::MaybeInitializeThreadSharding() {}

}  // namespace research_scann
