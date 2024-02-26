// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: scann/scann_ops/scann_assets.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_scann_2fscann_5fops_2fscann_5fassets_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_scann_2fscann_5fops_2fscann_5fassets_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3017000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3017003 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_scann_2fscann_5fops_2fscann_5fassets_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_scann_2fscann_5fops_2fscann_5fassets_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto;
namespace research_scann {
class ScannAsset;
struct ScannAssetDefaultTypeInternal;
extern ScannAssetDefaultTypeInternal _ScannAsset_default_instance_;
class ScannAssets;
struct ScannAssetsDefaultTypeInternal;
extern ScannAssetsDefaultTypeInternal _ScannAssets_default_instance_;
}  // namespace research_scann
PROTOBUF_NAMESPACE_OPEN
template<> ::research_scann::ScannAsset* Arena::CreateMaybeMessage<::research_scann::ScannAsset>(Arena*);
template<> ::research_scann::ScannAssets* Arena::CreateMaybeMessage<::research_scann::ScannAssets>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace research_scann {

enum ScannAsset_AssetType : int {
  ScannAsset_AssetType_UNSPECIFIED_TYPE = 0,
  ScannAsset_AssetType_DATASET = 1,
  ScannAsset_AssetType_INT8_DATASET = 2,
  ScannAsset_AssetType_AH_DATASET = 3,
  ScannAsset_AssetType_TOKENIZATION = 4,
  ScannAsset_AssetType_REORDERING_INT8_MULTIPLIERS = 5,
  ScannAsset_AssetType_BRUTE_FORCE_INT8_MULTIPLIERS = 6,
  ScannAsset_AssetType_AH_CENTERS = 7,
  ScannAsset_AssetType_PARTITIONER = 8,
  ScannAsset_AssetType_DATASET_NPY = 9,
  ScannAsset_AssetType_INT8_DATASET_NPY = 10,
  ScannAsset_AssetType_AH_DATASET_NPY = 11,
  ScannAsset_AssetType_TOKENIZATION_NPY = 12,
  ScannAsset_AssetType_INT8_MULTIPLIERS_NPY = 13,
  ScannAsset_AssetType_INT8_NORMS_NPY = 14
};
bool ScannAsset_AssetType_IsValid(int value);
constexpr ScannAsset_AssetType ScannAsset_AssetType_AssetType_MIN = ScannAsset_AssetType_UNSPECIFIED_TYPE;
constexpr ScannAsset_AssetType ScannAsset_AssetType_AssetType_MAX = ScannAsset_AssetType_INT8_NORMS_NPY;
constexpr int ScannAsset_AssetType_AssetType_ARRAYSIZE = ScannAsset_AssetType_AssetType_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* ScannAsset_AssetType_descriptor();
template<typename T>
inline const std::string& ScannAsset_AssetType_Name(T enum_t_value) {
  static_assert(::std::is_same<T, ScannAsset_AssetType>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function ScannAsset_AssetType_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    ScannAsset_AssetType_descriptor(), enum_t_value);
}
inline bool ScannAsset_AssetType_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, ScannAsset_AssetType* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<ScannAsset_AssetType>(
    ScannAsset_AssetType_descriptor(), name, value);
}
// ===================================================================

class ScannAsset final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:research_scann.ScannAsset) */ {
 public:
  inline ScannAsset() : ScannAsset(nullptr) {}
  ~ScannAsset() override;
  explicit constexpr ScannAsset(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ScannAsset(const ScannAsset& from);
  ScannAsset(ScannAsset&& from) noexcept
    : ScannAsset() {
    *this = ::std::move(from);
  }

  inline ScannAsset& operator=(const ScannAsset& from) {
    CopyFrom(from);
    return *this;
  }
  inline ScannAsset& operator=(ScannAsset&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const ScannAsset& default_instance() {
    return *internal_default_instance();
  }
  static inline const ScannAsset* internal_default_instance() {
    return reinterpret_cast<const ScannAsset*>(
               &_ScannAsset_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ScannAsset& a, ScannAsset& b) {
    a.Swap(&b);
  }
  inline void Swap(ScannAsset* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ScannAsset* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ScannAsset* New() const final {
    return new ScannAsset();
  }

  ScannAsset* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ScannAsset>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const ScannAsset& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const ScannAsset& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to, const ::PROTOBUF_NAMESPACE_ID::Message&from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ScannAsset* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "research_scann.ScannAsset";
  }
  protected:
  explicit ScannAsset(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef ScannAsset_AssetType AssetType;
  static constexpr AssetType UNSPECIFIED_TYPE =
    ScannAsset_AssetType_UNSPECIFIED_TYPE;
  static constexpr AssetType DATASET =
    ScannAsset_AssetType_DATASET;
  static constexpr AssetType INT8_DATASET =
    ScannAsset_AssetType_INT8_DATASET;
  static constexpr AssetType AH_DATASET =
    ScannAsset_AssetType_AH_DATASET;
  static constexpr AssetType TOKENIZATION =
    ScannAsset_AssetType_TOKENIZATION;
  static constexpr AssetType REORDERING_INT8_MULTIPLIERS =
    ScannAsset_AssetType_REORDERING_INT8_MULTIPLIERS;
  static constexpr AssetType BRUTE_FORCE_INT8_MULTIPLIERS =
    ScannAsset_AssetType_BRUTE_FORCE_INT8_MULTIPLIERS;
  static constexpr AssetType AH_CENTERS =
    ScannAsset_AssetType_AH_CENTERS;
  static constexpr AssetType PARTITIONER =
    ScannAsset_AssetType_PARTITIONER;
  static constexpr AssetType DATASET_NPY =
    ScannAsset_AssetType_DATASET_NPY;
  static constexpr AssetType INT8_DATASET_NPY =
    ScannAsset_AssetType_INT8_DATASET_NPY;
  static constexpr AssetType AH_DATASET_NPY =
    ScannAsset_AssetType_AH_DATASET_NPY;
  static constexpr AssetType TOKENIZATION_NPY =
    ScannAsset_AssetType_TOKENIZATION_NPY;
  static constexpr AssetType INT8_MULTIPLIERS_NPY =
    ScannAsset_AssetType_INT8_MULTIPLIERS_NPY;
  static constexpr AssetType INT8_NORMS_NPY =
    ScannAsset_AssetType_INT8_NORMS_NPY;
  static inline bool AssetType_IsValid(int value) {
    return ScannAsset_AssetType_IsValid(value);
  }
  static constexpr AssetType AssetType_MIN =
    ScannAsset_AssetType_AssetType_MIN;
  static constexpr AssetType AssetType_MAX =
    ScannAsset_AssetType_AssetType_MAX;
  static constexpr int AssetType_ARRAYSIZE =
    ScannAsset_AssetType_AssetType_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  AssetType_descriptor() {
    return ScannAsset_AssetType_descriptor();
  }
  template<typename T>
  static inline const std::string& AssetType_Name(T enum_t_value) {
    static_assert(::std::is_same<T, AssetType>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function AssetType_Name.");
    return ScannAsset_AssetType_Name(enum_t_value);
  }
  static inline bool AssetType_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      AssetType* value) {
    return ScannAsset_AssetType_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kAssetPathFieldNumber = 2,
    kAssetTypeFieldNumber = 1,
  };
  // optional string asset_path = 2;
  bool has_asset_path() const;
  private:
  bool _internal_has_asset_path() const;
  public:
  void clear_asset_path();
  const std::string& asset_path() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_asset_path(ArgT0&& arg0, ArgT... args);
  std::string* mutable_asset_path();
  PROTOBUF_MUST_USE_RESULT std::string* release_asset_path();
  void set_allocated_asset_path(std::string* asset_path);
  private:
  const std::string& _internal_asset_path() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_asset_path(const std::string& value);
  std::string* _internal_mutable_asset_path();
  public:

  // optional .research_scann.ScannAsset.AssetType asset_type = 1;
  bool has_asset_type() const;
  private:
  bool _internal_has_asset_type() const;
  public:
  void clear_asset_type();
  ::research_scann::ScannAsset_AssetType asset_type() const;
  void set_asset_type(::research_scann::ScannAsset_AssetType value);
  private:
  ::research_scann::ScannAsset_AssetType _internal_asset_type() const;
  void _internal_set_asset_type(::research_scann::ScannAsset_AssetType value);
  public:

  // @@protoc_insertion_point(class_scope:research_scann.ScannAsset)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr asset_path_;
  int asset_type_;
  friend struct ::TableStruct_scann_2fscann_5fops_2fscann_5fassets_2eproto;
};
// -------------------------------------------------------------------

class ScannAssets final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:research_scann.ScannAssets) */ {
 public:
  inline ScannAssets() : ScannAssets(nullptr) {}
  ~ScannAssets() override;
  explicit constexpr ScannAssets(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ScannAssets(const ScannAssets& from);
  ScannAssets(ScannAssets&& from) noexcept
    : ScannAssets() {
    *this = ::std::move(from);
  }

  inline ScannAssets& operator=(const ScannAssets& from) {
    CopyFrom(from);
    return *this;
  }
  inline ScannAssets& operator=(ScannAssets&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance);
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const ScannAssets& default_instance() {
    return *internal_default_instance();
  }
  static inline const ScannAssets* internal_default_instance() {
    return reinterpret_cast<const ScannAssets*>(
               &_ScannAssets_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(ScannAssets& a, ScannAssets& b) {
    a.Swap(&b);
  }
  inline void Swap(ScannAssets* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ScannAssets* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ScannAssets* New() const final {
    return new ScannAssets();
  }

  ScannAssets* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ScannAssets>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const ScannAssets& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const ScannAssets& from);
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to, const ::PROTOBUF_NAMESPACE_ID::Message&from);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* _InternalSerialize(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ScannAssets* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "research_scann.ScannAssets";
  }
  protected:
  explicit ScannAssets(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kAssetsFieldNumber = 1,
    kTrainedOnTheFlyFieldNumber = 2,
  };
  // repeated .research_scann.ScannAsset assets = 1;
  int assets_size() const;
  private:
  int _internal_assets_size() const;
  public:
  void clear_assets();
  ::research_scann::ScannAsset* mutable_assets(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::research_scann::ScannAsset >*
      mutable_assets();
  private:
  const ::research_scann::ScannAsset& _internal_assets(int index) const;
  ::research_scann::ScannAsset* _internal_add_assets();
  public:
  const ::research_scann::ScannAsset& assets(int index) const;
  ::research_scann::ScannAsset* add_assets();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::research_scann::ScannAsset >&
      assets() const;

  // optional bool trained_on_the_fly = 2 [default = true];
  bool has_trained_on_the_fly() const;
  private:
  bool _internal_has_trained_on_the_fly() const;
  public:
  void clear_trained_on_the_fly();
  bool trained_on_the_fly() const;
  void set_trained_on_the_fly(bool value);
  private:
  bool _internal_trained_on_the_fly() const;
  void _internal_set_trained_on_the_fly(bool value);
  public:

  // @@protoc_insertion_point(class_scope:research_scann.ScannAssets)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::research_scann::ScannAsset > assets_;
  bool trained_on_the_fly_;
  friend struct ::TableStruct_scann_2fscann_5fops_2fscann_5fassets_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ScannAsset

// optional .research_scann.ScannAsset.AssetType asset_type = 1;
inline bool ScannAsset::_internal_has_asset_type() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool ScannAsset::has_asset_type() const {
  return _internal_has_asset_type();
}
inline void ScannAsset::clear_asset_type() {
  asset_type_ = 0;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::research_scann::ScannAsset_AssetType ScannAsset::_internal_asset_type() const {
  return static_cast< ::research_scann::ScannAsset_AssetType >(asset_type_);
}
inline ::research_scann::ScannAsset_AssetType ScannAsset::asset_type() const {
  // @@protoc_insertion_point(field_get:research_scann.ScannAsset.asset_type)
  return _internal_asset_type();
}
inline void ScannAsset::_internal_set_asset_type(::research_scann::ScannAsset_AssetType value) {
  assert(::research_scann::ScannAsset_AssetType_IsValid(value));
  _has_bits_[0] |= 0x00000002u;
  asset_type_ = value;
}
inline void ScannAsset::set_asset_type(::research_scann::ScannAsset_AssetType value) {
  _internal_set_asset_type(value);
  // @@protoc_insertion_point(field_set:research_scann.ScannAsset.asset_type)
}

// optional string asset_path = 2;
inline bool ScannAsset::_internal_has_asset_path() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool ScannAsset::has_asset_path() const {
  return _internal_has_asset_path();
}
inline void ScannAsset::clear_asset_path() {
  asset_path_.ClearToEmpty();
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& ScannAsset::asset_path() const {
  // @@protoc_insertion_point(field_get:research_scann.ScannAsset.asset_path)
  return _internal_asset_path();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ScannAsset::set_asset_path(ArgT0&& arg0, ArgT... args) {
 _has_bits_[0] |= 0x00000001u;
 asset_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:research_scann.ScannAsset.asset_path)
}
inline std::string* ScannAsset::mutable_asset_path() {
  std::string* _s = _internal_mutable_asset_path();
  // @@protoc_insertion_point(field_mutable:research_scann.ScannAsset.asset_path)
  return _s;
}
inline const std::string& ScannAsset::_internal_asset_path() const {
  return asset_path_.Get();
}
inline void ScannAsset::_internal_set_asset_path(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  asset_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ScannAsset::_internal_mutable_asset_path() {
  _has_bits_[0] |= 0x00000001u;
  return asset_path_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ScannAsset::release_asset_path() {
  // @@protoc_insertion_point(field_release:research_scann.ScannAsset.asset_path)
  if (!_internal_has_asset_path()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return asset_path_.ReleaseNonDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ScannAsset::set_allocated_asset_path(std::string* asset_path) {
  if (asset_path != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  asset_path_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), asset_path,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:research_scann.ScannAsset.asset_path)
}

// -------------------------------------------------------------------

// ScannAssets

// repeated .research_scann.ScannAsset assets = 1;
inline int ScannAssets::_internal_assets_size() const {
  return assets_.size();
}
inline int ScannAssets::assets_size() const {
  return _internal_assets_size();
}
inline void ScannAssets::clear_assets() {
  assets_.Clear();
}
inline ::research_scann::ScannAsset* ScannAssets::mutable_assets(int index) {
  // @@protoc_insertion_point(field_mutable:research_scann.ScannAssets.assets)
  return assets_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::research_scann::ScannAsset >*
ScannAssets::mutable_assets() {
  // @@protoc_insertion_point(field_mutable_list:research_scann.ScannAssets.assets)
  return &assets_;
}
inline const ::research_scann::ScannAsset& ScannAssets::_internal_assets(int index) const {
  return assets_.Get(index);
}
inline const ::research_scann::ScannAsset& ScannAssets::assets(int index) const {
  // @@protoc_insertion_point(field_get:research_scann.ScannAssets.assets)
  return _internal_assets(index);
}
inline ::research_scann::ScannAsset* ScannAssets::_internal_add_assets() {
  return assets_.Add();
}
inline ::research_scann::ScannAsset* ScannAssets::add_assets() {
  ::research_scann::ScannAsset* _add = _internal_add_assets();
  // @@protoc_insertion_point(field_add:research_scann.ScannAssets.assets)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::research_scann::ScannAsset >&
ScannAssets::assets() const {
  // @@protoc_insertion_point(field_list:research_scann.ScannAssets.assets)
  return assets_;
}

// optional bool trained_on_the_fly = 2 [default = true];
inline bool ScannAssets::_internal_has_trained_on_the_fly() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool ScannAssets::has_trained_on_the_fly() const {
  return _internal_has_trained_on_the_fly();
}
inline void ScannAssets::clear_trained_on_the_fly() {
  trained_on_the_fly_ = true;
  _has_bits_[0] &= ~0x00000001u;
}
inline bool ScannAssets::_internal_trained_on_the_fly() const {
  return trained_on_the_fly_;
}
inline bool ScannAssets::trained_on_the_fly() const {
  // @@protoc_insertion_point(field_get:research_scann.ScannAssets.trained_on_the_fly)
  return _internal_trained_on_the_fly();
}
inline void ScannAssets::_internal_set_trained_on_the_fly(bool value) {
  _has_bits_[0] |= 0x00000001u;
  trained_on_the_fly_ = value;
}
inline void ScannAssets::set_trained_on_the_fly(bool value) {
  _internal_set_trained_on_the_fly(value);
  // @@protoc_insertion_point(field_set:research_scann.ScannAssets.trained_on_the_fly)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace research_scann

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::research_scann::ScannAsset_AssetType> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::research_scann::ScannAsset_AssetType>() {
  return ::research_scann::ScannAsset_AssetType_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_scann_2fscann_5fops_2fscann_5fassets_2eproto
