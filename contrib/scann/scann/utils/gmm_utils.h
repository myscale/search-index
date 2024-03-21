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



#ifndef SCANN_UTILS_GMM_UTILS_H_
#define SCANN_UTILS_GMM_UTILS_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "absl/time/time.h"
#include "scann/data_format/dataset.h"
#include "scann/distance_measures/distance_measure_base.h"
#include "scann/oss_wrappers/scann_random.h"
#include "scann/oss_wrappers/scann_threadpool.h"
#include "scann/proto/partitioning.pb.h"
#include "scann/utils/types.h"

namespace research_scann {

class GmmUtilsImplInterface;

class GmmUtils {
 public:
  struct Options {
    int32_t seed = kDeterministicSeed;

    int32_t max_iterations = 10;

    double epsilon = 1e-5;

    absl::Duration max_iteration_duration = absl::InfiniteDuration();

    int32_t min_cluster_size = 1;

    int32_t max_cluster_size = std::numeric_limits<int32_t>::max();

    double perturbation = 1e-7;

    shared_ptr<ThreadPool> parallelization_pool;

    enum PartitionAssignmentType {
      UNBALANCED,

      GREEDY_BALANCED,

      MIN_COST_MAX_FLOW,
    };

    PartitionAssignmentType partition_assignment_type = UNBALANCED;

    enum CenterReassignmentType {
      RANDOM_REASSIGNMENT,

      SPLIT_LARGEST_CLUSTERS,

      PCA_SPLITTING,
    };

    CenterReassignmentType center_reassignment_type = RANDOM_REASSIGNMENT;

    enum CenterInitializationType {
      MEAN_DISTANCE_INITIALIZATION,

      KMEANS_PLUS_PLUS,

      RANDOM_INITIALIZATION,
    };

    CenterInitializationType center_initialization_type = KMEANS_PLUS_PLUS;

    int32_t max_power_of_2_split = 1;
  };

  GmmUtils(shared_ptr<const DistanceMeasure> dist, Options opts);

  explicit GmmUtils(shared_ptr<const DistanceMeasure> dist)
      : GmmUtils(std::move(dist), Options()) {}

  struct ComputeKmeansClusteringOptions {
    ConstSpan<DatapointIndex> subset;

    vector<vector<DatapointIndex>>* final_partitions = nullptr;

    bool spherical = false;

    ConstSpan<float> weights;

    // if true, do one pass assignment only
    bool one_pass_assignment_only = false;
  };

  Status ComputeKmeansClustering(
      const Dataset& dataset, const int32_t num_clusters,
      DenseDataset<double>* final_centers,
      const ComputeKmeansClusteringOptions& kmeans_opts);

  Status ComputeKmeansClustering(const Dataset& dataset,
                                 const int32_t num_clusters,
                                 DenseDataset<double>* final_centers) {
    ComputeKmeansClusteringOptions kmeans_opts;
    return ComputeKmeansClustering(dataset, num_clusters, final_centers,
                                   kmeans_opts);
  }

  StatusOr<double> ComputeSpillingThreshold(
      const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      const DenseDataset<double>& centers,
      const DatabaseSpillingConfig::SpillingType spilling_type,
      const float total_spill_factor, const uint32_t max_centers);

  Status RecomputeCentroidsSimple(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids) const;

  Status RecomputeCentroidsWeighted(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<float> weights, bool spherical,
      DenseDataset<double>* centroids) const;

  Status InitializeCenters(const Dataset& dataset,
                           ConstSpan<DatapointIndex> subset,
                           int32_t num_clusters, ConstSpan<float> weights,
                           DenseDataset<double>* initial_centers);

  using PartitionAssignmentFn =
      std::function<vector<pair<DatapointIndex, double>>(
          GmmUtilsImplInterface* impl, const DistanceMeasure& distance,
          const DenseDataset<double>& center, ThreadPool* pool)>;

 private:
  SCANN_OUTLINE Status KMeansImpl(
      bool spherical, const Dataset& dataset, ConstSpan<DatapointIndex> subset,
      int32_t num_clusters, PartitionAssignmentFn partition_assignment_fn,
      DenseDataset<double>* final_centers,
      vector<vector<DatapointIndex>>* final_partitions);

  Status RandomReinitializeCenters(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids,
      std::vector<double>* convergence_means);

  Status SplitLargeClusterReinitialization(
      ConstSpan<uint32_t> partition_sizes, bool spherical,
      DenseDataset<double>* centroids, std::vector<double>* convergence_means);

  Status PCAKmeansReinitialization(
      ConstSpan<pair<uint32_t, double>> top1_results,
      GmmUtilsImplInterface* impl, ConstSpan<uint32_t> partition_sizes,
      bool spherical, DenseDataset<double>* centroids,
      std::vector<double>* convergence_means) const;

  Status MeanDistanceInitializeCenters(const Dataset& dataset,
                                       ConstSpan<DatapointIndex> subset,
                                       int32_t num_clusters,
                                       ConstSpan<float> weights,
                                       DenseDataset<double>* initial_centers);
  Status KMeansPPInitializeCenters(const Dataset& dataset,
                                   ConstSpan<DatapointIndex> subset,
                                   int32_t num_clusters,
                                   ConstSpan<float> weights,
                                   DenseDataset<double>* initial_centers);
  Status RandomInitializeCenters(const Dataset& dataset,
                                 ConstSpan<DatapointIndex> subset,
                                 int32_t num_clusters, ConstSpan<float> weights,
                                 DenseDataset<double>* initial_centers);

  shared_ptr<const DistanceMeasure> distance_;
  Options opts_;
  MTRandom random_;
};

}  // namespace research_scann

#endif
