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

#include "scann/hashes/asymmetric_hashing2/querying.h"

#include <cstdint>
#include <utility>

#include "scann/utils/common.h"
#include "scann/utils/intrinsics/flags.h"

using std::shared_ptr;

namespace research_scann {
namespace asymmetric_hashing2 {

PackedDataset CreatePackedDataset(
    const DenseDataset<uint8_t>& hashed_database,
    DenseDataset<uint8_t>* packed_database,
    size_t num_blocks) {
  PackedDataset result;
  result.num_datapoints = hashed_database.size();
  result.num_blocks = num_blocks;
  if (hashed_database.empty()) return result;

  if (hashed_database.hash_4bit)
    CHECK_EQ(hashed_database.dimensionality(), (num_blocks + 1) / 2);
  else
    CHECK_EQ(hashed_database.dimensionality(), num_blocks);

  auto packed_size = num_blocks * ((hashed_database.size() + 31) & (~31)) / 2;
  if (!packed_database) {
    // allocate new array for bit_packed_data
    result.bit_packed_vector.resize(packed_size);
    result.bit_packed_data = absl::MakeSpan(result.bit_packed_vector);
  }
  else {
    size_t data_size = packed_database->size() * packed_database->dimensionality();
    CHECK_EQ(data_size, packed_size);
    result.bit_packed_data = absl::Span<uint8_t>(
      &packed_database->mutable_data()[0], data_size);
  }

  if (hashed_database.hash_4bit)
    asymmetric_hashing_internal::CreatePackedDataset<true>(
      result.bit_packed_data, hashed_database, num_blocks);
  else
    asymmetric_hashing_internal::CreatePackedDataset<false>(
      result.bit_packed_data, hashed_database, num_blocks);
  VLOG(1) << "Packing dataset num_data=" << hashed_database.size()
          << " num_blocks=" << num_blocks
          << " hash_4bit=" << hashed_database.hash_4bit
          << " before_size=" << hashed_database.data().size()
          << " after_size=" << result.bit_packed_data.size();

  return result;
}

DenseDataset<uint8_t> UnpackDataset(const PackedDataset& packed) {
  const size_t num_dim = packed.num_blocks, num_dp = packed.num_datapoints;
  const size_t packed_dim = (num_dim + 1) / 2;
  VLOG(1) << "UnpackDataset, num_dim=" << num_dim
          << " num_dp=" << num_dp
          << " packed_dim=" << packed_dim;

  vector<uint8_t> unpacked(packed_dim * num_dp, 0);
  auto set_unpacked = [&unpacked, packed_dim](int row, int offset, uint8_t value) {
    // if offset % 2 == 1, shift by 4 bits
    DCHECK_LT(row * packed_dim + (offset >> 1), unpacked.size());
    unpacked[row * packed_dim + (offset >> 1)] |= value << ((offset & 1) << 2);
  };

  int idx = 0;
  for (int dp_block = 0; dp_block < num_dp / 32; dp_block++) {
    const int out_idx = 32 * dp_block;
    for (int dim = 0; dim < num_dim; dim++) {
      for (int offset = 0; offset < 16; offset++) {
        uint8_t data = packed.bit_packed_data[idx++];
        // unpacked[(out_idx | offset) * num_dim + dim] = data & 15;
        // unpacked[(out_idx | 16 | offset) * num_dim + dim] = data >> 4;
        set_unpacked(out_idx | offset, dim, data & 15);
        set_unpacked(out_idx | 16 | offset, dim, data >> 4);
      }
    }
  }

  if (num_dp % 32 != 0) {
    const int out_idx = num_dp - (num_dp % 32);
    for (int dim = 0; dim < num_dim; dim++) {
      for (int offset = 0; offset < 16; offset++) {
        uint8_t data = packed.bit_packed_data[idx++];
        int idx1 = out_idx | offset, idx2 = out_idx | 16 | offset;
        // if (idx1 < num_dp) unpacked[idx1 * num_dim + dim] = data & 15;
        // if (idx2 < num_dp) unpacked[idx2 * num_dim + dim] = data >> 4;
        if (idx1 < num_dp) set_unpacked(idx1, dim, data & 15);
        if (idx2 < num_dp) set_unpacked(idx2, dim, data >> 4);
      }
    }
  }
  return DenseDataset<uint8_t>(unpacked, packed.num_datapoints);
}

template <typename T>
AsymmetricQueryer<T>::AsymmetricQueryer(
    shared_ptr<const ChunkingProjection<T>> projector,
    shared_ptr<const DistanceMeasure> lookup_distance,
    shared_ptr<const Model<T>> model)
    : projector_(std::move(projector)),
      lookup_distance_(std::move(lookup_distance)),
      model_(std::move(model)) {}

template <typename T>
StatusOr<LookupTable> AsymmetricQueryer<T>::CreateLookupTable(
    const DatapointPtr<T>& query,
    AsymmetricHasherConfig::LookupType lookup_type,
    AsymmetricHasherConfig::FixedPointLUTConversionOptions
        float_int_conversion_options) const {
  switch (lookup_type) {
    case AsymmetricHasherConfig::FLOAT:
      return CreateLookupTable<float>(query, float_int_conversion_options);
    case AsymmetricHasherConfig::INT8:
    case AsymmetricHasherConfig::INT8_LUT16:
      return CreateLookupTable<int8_t>(query, float_int_conversion_options);
    case AsymmetricHasherConfig::INT16:
      return CreateLookupTable<int16_t>(query, float_int_conversion_options);
    default:
      return InvalidArgumentError("Unrecognized lookup type.");
  }
}

SCANN_INSTANTIATE_TYPED_CLASS(, AsymmetricQueryer);

}  // namespace asymmetric_hashing2
}  // namespace research_scann
