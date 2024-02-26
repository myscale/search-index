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



#ifndef SCANN_BASE_RESTRICT_ALLOWLIST_H_
#define SCANN_BASE_RESTRICT_ALLOWLIST_H_

#include <cstdint>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "scann/oss_wrappers/scann_bits.h"
#include "scann/utils/bit_iterator.h"
#include "scann/utils/types.h"
#include "tensorflow/core/platform/macros.h"
#include "absl/numeric/int128.h"
#include <SearchIndex/Common/DenseBitmap.h>

namespace research_scann {

class RestrictTokenMap;
class RestrictAllowlistConstView;

class RestrictAllowlist {
 public:
  RestrictAllowlist(DatapointIndex num_points, bool default_whitelisted);
  RestrictAllowlist() : RestrictAllowlist(0, false) {}
  RestrictAllowlist(Search::DenseBitmap* id_filter,
                    ConstSpan<Search::idx_t> id_list):
      num_points_(id_list.size()), id_filter_(id_filter), id_list_(id_list) {
  }

  ~RestrictAllowlist();

  RestrictAllowlist(std::vector<size_t>&& allowlist_array,
                    DatapointIndex num_points, bool default_whitelisted);

  RestrictAllowlist(const RestrictAllowlist& rhs);
  RestrictAllowlist(RestrictAllowlist&& rhs) noexcept = default;
  RestrictAllowlist& operator=(const RestrictAllowlist& rhs);
  RestrictAllowlist& operator=(RestrictAllowlist&& rhs) = default;

  explicit RestrictAllowlist(const RestrictAllowlistConstView& view);

  void Initialize(DatapointIndex num_points, bool default_whitelisted);

  RestrictAllowlist CopyWithCapacity(
      DatapointIndex capacity,
      vector<size_t>&& backing_storage = vector<size_t>()) const;

  RestrictAllowlist CopyWithSize(
      DatapointIndex size, bool default_whitelisted,
      vector<size_t>&& backing_storage = vector<size_t>()) const;

  void Append(bool is_whitelisted);

  void Resize(size_t num_points, bool default_whitelisted);

  bool CapacityAvailableForAppend(DatapointIndex dp_index) const;
  bool CapacityAvailableForAppend() const {
    return CapacityAvailableForAppend(num_points_);
  }

  bool IsWhitelisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return IsWhitelistedNoBoundsCheck(dp_index);
  }

  bool IsWhitelistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsWhitelisted(dp_index);
  }

  void set_allowlist_recycling_fn(
      std::function<void(std::vector<size_t>&&)> f) {
    allowlist_recycling_fn_ = std::move(f);
  }

  // might not be effective for small number of whitelisted points
  DatapointIndex NumPointsWhitelisted() const { return num_points_; }

  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }

  using Iterator = BitIterator<ConstSpan<size_t>, DatapointIndex>;

  // Only works for DirectList:
  // use RestrictAllowlistConstView to materialize the indirect list
  Iterator WhitelistedPointIterator() const {
    CHECK(isDirectList()) << "Indirect list doesn't support WhitelistedPointIterator size=" << size();
    return Iterator(allowlist_array_);
  }

  bool isDirectList() const {
    return id_filter_ == nullptr && id_list_.empty();
  }

  ConstSpan<Search::idx_t> getIDList() const {
    return id_list_;
  }

  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    CHECK(isDirectList());
    return _data()[dp_index / kBitsPerWord];
  }

  const size_t* _data() const {
      return allowlist_array_.data();
  }

  static uint8_t FindLSBSetNonZeroNative(size_t word) {
    static_assert(sizeof(word) == 8 || sizeof(word) == 4, "");
    return (sizeof(word) == 8) ? bits::FindLSBSetNonZero64(word)
                               : bits::FindLSBSetNonZero(word);
  }

  static constexpr size_t kBitsPerWord = sizeof(size_t) * CHAR_BIT;

  static constexpr size_t kOne = 1;

  static constexpr size_t kZero = 0;

  static constexpr size_t kAllOnes = ~kZero;

  static constexpr size_t kRoundDownMask = ~(kBitsPerWord - 1);

 private:
  bool IsWhitelistedNoBoundsCheck(DatapointIndex dp_index) const {
    return GetWordContainingDatapoint(dp_index) &
           (kOne << (dp_index % kBitsPerWord));
  }

  template <typename Lambda>
  void PointwiseLogic(const RestrictAllowlistConstView& rhs, Lambda lambda,
                      bool zero_trailing);

  // either allowlist_array_ or id_filter_/id_list_ can be non-empty
  std::vector<size_t> allowlist_array_;
  Search::DenseBitmap* id_filter_{nullptr};
  ConstSpan<Search::idx_t> id_list_;

  DatapointIndex num_points_;

  std::function<void(std::vector<size_t>&&)> allowlist_recycling_fn_;

  friend class RestrictTokenMap;

  friend class RestrictAllowlistConstView;
};

class DummyAllowlist {
 public:
  explicit DummyAllowlist(DatapointIndex num_points);

  bool IsAllowlisted(DatapointIndex dp_index) const { return true; }

  class Iterator {
   public:
    DatapointIndex value() const { return value_; }
    bool Done() const { return value_ >= num_points_; }
    void Next() { ++value_; }

   private:
    friend class DummyAllowlist;

    explicit Iterator(DatapointIndex num_points);

    DatapointIndex value_;

    DatapointIndex num_points_;
  };

  Iterator AllowlistedPointIterator() const { return Iterator(num_points_); }

 private:
  DatapointIndex num_points_;
};

class RestrictAllowlistConstView {
 public:
  RestrictAllowlistConstView() {}

  explicit RestrictAllowlistConstView(const RestrictAllowlist& whitelist)
      : allowlist_array_(whitelist.allowlist_array_.data()),
        num_points_(whitelist.num_points_) {}

  explicit RestrictAllowlistConstView(const RestrictAllowlist* whitelist)
      : allowlist_array_(whitelist ? whitelist->allowlist_array_.data()
                                   : nullptr),
        num_points_(whitelist ? whitelist->num_points_ : 0) {}

  RestrictAllowlistConstView(ConstSpan<size_t> storage,
                             DatapointIndex num_points)
      : allowlist_array_(storage.data()), num_points_(num_points) {
    DCHECK_EQ(storage.size(),
              DivRoundUp(num_points, RestrictAllowlist::kBitsPerWord));
  }

  explicit RestrictAllowlistConstView(const RestrictAllowlist& whitelist,
                                      const ConstSpan<uint32_t>& index_to_filter_pos)
      : allowlist_array_(whitelist.allowlist_array_.data()),
        id_filter_(whitelist.id_filter_),
        index_to_filter_pos_(index_to_filter_pos),
        num_points_(index_to_filter_pos.size()) {}

  bool IsWhitelisted(DatapointIndex dp_index) const {
    DCHECK_LT(dp_index, num_points_);
    return GetWordContainingDatapoint(dp_index) &
           (RestrictAllowlist::kOne
            << (dp_index % RestrictAllowlist::kBitsPerWord));
  }

  bool IsWhitelistedWithDefault(DatapointIndex dp_index,
                                bool default_value) const {
    if (dp_index >= num_points()) return default_value;
    return IsWhitelisted(dp_index);
  }

  size_t GetWordContainingDatapoint(DatapointIndex dp_index) const {
    // TODO how many times is it called
    return data()[dp_index / RestrictAllowlist::kBitsPerWord];
  }

  bool isDirectList() const {
    return index_to_filter_pos_.empty() && id_filter_ == nullptr;
  }

  const size_t* data() const;

  DatapointIndex num_points() const { return num_points_; }
  DatapointIndex size() const { return num_points_; }
  bool empty() const { return !allowlist_array_; }

  operator bool() const { return !empty(); }

 private:
  const size_t* allowlist_array_ = nullptr;
  Search::DenseBitmap* id_filter_ = nullptr;
  ConstSpan<uint32_t> index_to_filter_pos_;
  mutable std::vector<size_t> materialized_allowlist_;
  DatapointIndex num_points_ = 0;
  friend class RestrictAllowlist;
};

}  // namespace research_scann

#endif
