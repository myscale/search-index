// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/tensor_bundle.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto

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
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_slice.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto;
namespace tensorflow {
class BundleEntryProto;
struct BundleEntryProtoDefaultTypeInternal;
extern BundleEntryProtoDefaultTypeInternal _BundleEntryProto_default_instance_;
class BundleHeaderProto;
struct BundleHeaderProtoDefaultTypeInternal;
extern BundleHeaderProtoDefaultTypeInternal _BundleHeaderProto_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::BundleEntryProto* Arena::CreateMaybeMessage<::tensorflow::BundleEntryProto>(Arena*);
template<> ::tensorflow::BundleHeaderProto* Arena::CreateMaybeMessage<::tensorflow::BundleHeaderProto>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

enum BundleHeaderProto_Endianness : int {
  BundleHeaderProto_Endianness_LITTLE = 0,
  BundleHeaderProto_Endianness_BIG = 1,
  BundleHeaderProto_Endianness_BundleHeaderProto_Endianness_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::min(),
  BundleHeaderProto_Endianness_BundleHeaderProto_Endianness_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<::PROTOBUF_NAMESPACE_ID::int32>::max()
};
bool BundleHeaderProto_Endianness_IsValid(int value);
constexpr BundleHeaderProto_Endianness BundleHeaderProto_Endianness_Endianness_MIN = BundleHeaderProto_Endianness_LITTLE;
constexpr BundleHeaderProto_Endianness BundleHeaderProto_Endianness_Endianness_MAX = BundleHeaderProto_Endianness_BIG;
constexpr int BundleHeaderProto_Endianness_Endianness_ARRAYSIZE = BundleHeaderProto_Endianness_Endianness_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* BundleHeaderProto_Endianness_descriptor();
template<typename T>
inline const std::string& BundleHeaderProto_Endianness_Name(T enum_t_value) {
  static_assert(::std::is_same<T, BundleHeaderProto_Endianness>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function BundleHeaderProto_Endianness_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    BundleHeaderProto_Endianness_descriptor(), enum_t_value);
}
inline bool BundleHeaderProto_Endianness_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, BundleHeaderProto_Endianness* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<BundleHeaderProto_Endianness>(
    BundleHeaderProto_Endianness_descriptor(), name, value);
}
// ===================================================================

class BundleHeaderProto final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.BundleHeaderProto) */ {
 public:
  inline BundleHeaderProto() : BundleHeaderProto(nullptr) {}
  ~BundleHeaderProto() override;
  explicit constexpr BundleHeaderProto(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  BundleHeaderProto(const BundleHeaderProto& from);
  BundleHeaderProto(BundleHeaderProto&& from) noexcept
    : BundleHeaderProto() {
    *this = ::std::move(from);
  }

  inline BundleHeaderProto& operator=(const BundleHeaderProto& from) {
    CopyFrom(from);
    return *this;
  }
  inline BundleHeaderProto& operator=(BundleHeaderProto&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
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
  static const BundleHeaderProto& default_instance() {
    return *internal_default_instance();
  }
  static inline const BundleHeaderProto* internal_default_instance() {
    return reinterpret_cast<const BundleHeaderProto*>(
               &_BundleHeaderProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(BundleHeaderProto& a, BundleHeaderProto& b) {
    a.Swap(&b);
  }
  inline void Swap(BundleHeaderProto* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(BundleHeaderProto* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline BundleHeaderProto* New() const final {
    return new BundleHeaderProto();
  }

  BundleHeaderProto* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<BundleHeaderProto>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const BundleHeaderProto& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const BundleHeaderProto& from);
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
  void InternalSwap(BundleHeaderProto* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.BundleHeaderProto";
  }
  protected:
  explicit BundleHeaderProto(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef BundleHeaderProto_Endianness Endianness;
  static constexpr Endianness LITTLE =
    BundleHeaderProto_Endianness_LITTLE;
  static constexpr Endianness BIG =
    BundleHeaderProto_Endianness_BIG;
  static inline bool Endianness_IsValid(int value) {
    return BundleHeaderProto_Endianness_IsValid(value);
  }
  static constexpr Endianness Endianness_MIN =
    BundleHeaderProto_Endianness_Endianness_MIN;
  static constexpr Endianness Endianness_MAX =
    BundleHeaderProto_Endianness_Endianness_MAX;
  static constexpr int Endianness_ARRAYSIZE =
    BundleHeaderProto_Endianness_Endianness_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  Endianness_descriptor() {
    return BundleHeaderProto_Endianness_descriptor();
  }
  template<typename T>
  static inline const std::string& Endianness_Name(T enum_t_value) {
    static_assert(::std::is_same<T, Endianness>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function Endianness_Name.");
    return BundleHeaderProto_Endianness_Name(enum_t_value);
  }
  static inline bool Endianness_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      Endianness* value) {
    return BundleHeaderProto_Endianness_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kVersionFieldNumber = 3,
    kNumShardsFieldNumber = 1,
    kEndiannessFieldNumber = 2,
  };
  // .tensorflow.VersionDef version = 3;
  bool has_version() const;
  private:
  bool _internal_has_version() const;
  public:
  void clear_version();
  const ::tensorflow::VersionDef& version() const;
  PROTOBUF_MUST_USE_RESULT ::tensorflow::VersionDef* release_version();
  ::tensorflow::VersionDef* mutable_version();
  void set_allocated_version(::tensorflow::VersionDef* version);
  private:
  const ::tensorflow::VersionDef& _internal_version() const;
  ::tensorflow::VersionDef* _internal_mutable_version();
  public:
  void unsafe_arena_set_allocated_version(
      ::tensorflow::VersionDef* version);
  ::tensorflow::VersionDef* unsafe_arena_release_version();

  // int32 num_shards = 1;
  void clear_num_shards();
  ::PROTOBUF_NAMESPACE_ID::int32 num_shards() const;
  void set_num_shards(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_num_shards() const;
  void _internal_set_num_shards(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // .tensorflow.BundleHeaderProto.Endianness endianness = 2;
  void clear_endianness();
  ::tensorflow::BundleHeaderProto_Endianness endianness() const;
  void set_endianness(::tensorflow::BundleHeaderProto_Endianness value);
  private:
  ::tensorflow::BundleHeaderProto_Endianness _internal_endianness() const;
  void _internal_set_endianness(::tensorflow::BundleHeaderProto_Endianness value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.BundleHeaderProto)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::tensorflow::VersionDef* version_;
  ::PROTOBUF_NAMESPACE_ID::int32 num_shards_;
  int endianness_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto;
};
// -------------------------------------------------------------------

class BundleEntryProto final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.BundleEntryProto) */ {
 public:
  inline BundleEntryProto() : BundleEntryProto(nullptr) {}
  ~BundleEntryProto() override;
  explicit constexpr BundleEntryProto(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  BundleEntryProto(const BundleEntryProto& from);
  BundleEntryProto(BundleEntryProto&& from) noexcept
    : BundleEntryProto() {
    *this = ::std::move(from);
  }

  inline BundleEntryProto& operator=(const BundleEntryProto& from) {
    CopyFrom(from);
    return *this;
  }
  inline BundleEntryProto& operator=(BundleEntryProto&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
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
  static const BundleEntryProto& default_instance() {
    return *internal_default_instance();
  }
  static inline const BundleEntryProto* internal_default_instance() {
    return reinterpret_cast<const BundleEntryProto*>(
               &_BundleEntryProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(BundleEntryProto& a, BundleEntryProto& b) {
    a.Swap(&b);
  }
  inline void Swap(BundleEntryProto* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(BundleEntryProto* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline BundleEntryProto* New() const final {
    return new BundleEntryProto();
  }

  BundleEntryProto* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<BundleEntryProto>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const BundleEntryProto& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const BundleEntryProto& from);
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
  void InternalSwap(BundleEntryProto* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.BundleEntryProto";
  }
  protected:
  explicit BundleEntryProto(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kSlicesFieldNumber = 7,
    kShapeFieldNumber = 2,
    kDtypeFieldNumber = 1,
    kShardIdFieldNumber = 3,
    kOffsetFieldNumber = 4,
    kSizeFieldNumber = 5,
    kCrc32CFieldNumber = 6,
  };
  // repeated .tensorflow.TensorSliceProto slices = 7;
  int slices_size() const;
  private:
  int _internal_slices_size() const;
  public:
  void clear_slices();
  ::tensorflow::TensorSliceProto* mutable_slices(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::TensorSliceProto >*
      mutable_slices();
  private:
  const ::tensorflow::TensorSliceProto& _internal_slices(int index) const;
  ::tensorflow::TensorSliceProto* _internal_add_slices();
  public:
  const ::tensorflow::TensorSliceProto& slices(int index) const;
  ::tensorflow::TensorSliceProto* add_slices();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::TensorSliceProto >&
      slices() const;

  // .tensorflow.TensorShapeProto shape = 2;
  bool has_shape() const;
  private:
  bool _internal_has_shape() const;
  public:
  void clear_shape();
  const ::tensorflow::TensorShapeProto& shape() const;
  PROTOBUF_MUST_USE_RESULT ::tensorflow::TensorShapeProto* release_shape();
  ::tensorflow::TensorShapeProto* mutable_shape();
  void set_allocated_shape(::tensorflow::TensorShapeProto* shape);
  private:
  const ::tensorflow::TensorShapeProto& _internal_shape() const;
  ::tensorflow::TensorShapeProto* _internal_mutable_shape();
  public:
  void unsafe_arena_set_allocated_shape(
      ::tensorflow::TensorShapeProto* shape);
  ::tensorflow::TensorShapeProto* unsafe_arena_release_shape();

  // .tensorflow.DataType dtype = 1;
  void clear_dtype();
  ::tensorflow::DataType dtype() const;
  void set_dtype(::tensorflow::DataType value);
  private:
  ::tensorflow::DataType _internal_dtype() const;
  void _internal_set_dtype(::tensorflow::DataType value);
  public:

  // int32 shard_id = 3;
  void clear_shard_id();
  ::PROTOBUF_NAMESPACE_ID::int32 shard_id() const;
  void set_shard_id(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_shard_id() const;
  void _internal_set_shard_id(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int64 offset = 4;
  void clear_offset();
  ::PROTOBUF_NAMESPACE_ID::int64 offset() const;
  void set_offset(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_offset() const;
  void _internal_set_offset(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 size = 5;
  void clear_size();
  ::PROTOBUF_NAMESPACE_ID::int64 size() const;
  void set_size(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_size() const;
  void _internal_set_size(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // fixed32 crc32c = 6;
  void clear_crc32c();
  ::PROTOBUF_NAMESPACE_ID::uint32 crc32c() const;
  void set_crc32c(::PROTOBUF_NAMESPACE_ID::uint32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::uint32 _internal_crc32c() const;
  void _internal_set_crc32c(::PROTOBUF_NAMESPACE_ID::uint32 value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.BundleEntryProto)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::TensorSliceProto > slices_;
  ::tensorflow::TensorShapeProto* shape_;
  int dtype_;
  ::PROTOBUF_NAMESPACE_ID::int32 shard_id_;
  ::PROTOBUF_NAMESPACE_ID::int64 offset_;
  ::PROTOBUF_NAMESPACE_ID::int64 size_;
  ::PROTOBUF_NAMESPACE_ID::uint32 crc32c_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// BundleHeaderProto

// int32 num_shards = 1;
inline void BundleHeaderProto::clear_num_shards() {
  num_shards_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BundleHeaderProto::_internal_num_shards() const {
  return num_shards_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BundleHeaderProto::num_shards() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleHeaderProto.num_shards)
  return _internal_num_shards();
}
inline void BundleHeaderProto::_internal_set_num_shards(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  num_shards_ = value;
}
inline void BundleHeaderProto::set_num_shards(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_num_shards(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleHeaderProto.num_shards)
}

// .tensorflow.BundleHeaderProto.Endianness endianness = 2;
inline void BundleHeaderProto::clear_endianness() {
  endianness_ = 0;
}
inline ::tensorflow::BundleHeaderProto_Endianness BundleHeaderProto::_internal_endianness() const {
  return static_cast< ::tensorflow::BundleHeaderProto_Endianness >(endianness_);
}
inline ::tensorflow::BundleHeaderProto_Endianness BundleHeaderProto::endianness() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleHeaderProto.endianness)
  return _internal_endianness();
}
inline void BundleHeaderProto::_internal_set_endianness(::tensorflow::BundleHeaderProto_Endianness value) {
  
  endianness_ = value;
}
inline void BundleHeaderProto::set_endianness(::tensorflow::BundleHeaderProto_Endianness value) {
  _internal_set_endianness(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleHeaderProto.endianness)
}

// .tensorflow.VersionDef version = 3;
inline bool BundleHeaderProto::_internal_has_version() const {
  return this != internal_default_instance() && version_ != nullptr;
}
inline bool BundleHeaderProto::has_version() const {
  return _internal_has_version();
}
inline const ::tensorflow::VersionDef& BundleHeaderProto::_internal_version() const {
  const ::tensorflow::VersionDef* p = version_;
  return p != nullptr ? *p : reinterpret_cast<const ::tensorflow::VersionDef&>(
      ::tensorflow::_VersionDef_default_instance_);
}
inline const ::tensorflow::VersionDef& BundleHeaderProto::version() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleHeaderProto.version)
  return _internal_version();
}
inline void BundleHeaderProto::unsafe_arena_set_allocated_version(
    ::tensorflow::VersionDef* version) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(version_);
  }
  version_ = version;
  if (version) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.BundleHeaderProto.version)
}
inline ::tensorflow::VersionDef* BundleHeaderProto::release_version() {
  
  ::tensorflow::VersionDef* temp = version_;
  version_ = nullptr;
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  auto* old =  reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(temp);
  temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  if (GetArenaForAllocation() == nullptr) { delete old; }
#else  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (GetArenaForAllocation() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
  return temp;
}
inline ::tensorflow::VersionDef* BundleHeaderProto::unsafe_arena_release_version() {
  // @@protoc_insertion_point(field_release:tensorflow.BundleHeaderProto.version)
  
  ::tensorflow::VersionDef* temp = version_;
  version_ = nullptr;
  return temp;
}
inline ::tensorflow::VersionDef* BundleHeaderProto::_internal_mutable_version() {
  
  if (version_ == nullptr) {
    auto* p = CreateMaybeMessage<::tensorflow::VersionDef>(GetArenaForAllocation());
    version_ = p;
  }
  return version_;
}
inline ::tensorflow::VersionDef* BundleHeaderProto::mutable_version() {
  ::tensorflow::VersionDef* _msg = _internal_mutable_version();
  // @@protoc_insertion_point(field_mutable:tensorflow.BundleHeaderProto.version)
  return _msg;
}
inline void BundleHeaderProto::set_allocated_version(::tensorflow::VersionDef* version) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(version_);
  }
  if (version) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<
            ::PROTOBUF_NAMESPACE_ID::MessageLite>::GetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(version));
    if (message_arena != submessage_arena) {
      version = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, version, submessage_arena);
    }
    
  } else {
    
  }
  version_ = version;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.BundleHeaderProto.version)
}

// -------------------------------------------------------------------

// BundleEntryProto

// .tensorflow.DataType dtype = 1;
inline void BundleEntryProto::clear_dtype() {
  dtype_ = 0;
}
inline ::tensorflow::DataType BundleEntryProto::_internal_dtype() const {
  return static_cast< ::tensorflow::DataType >(dtype_);
}
inline ::tensorflow::DataType BundleEntryProto::dtype() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.dtype)
  return _internal_dtype();
}
inline void BundleEntryProto::_internal_set_dtype(::tensorflow::DataType value) {
  
  dtype_ = value;
}
inline void BundleEntryProto::set_dtype(::tensorflow::DataType value) {
  _internal_set_dtype(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleEntryProto.dtype)
}

// .tensorflow.TensorShapeProto shape = 2;
inline bool BundleEntryProto::_internal_has_shape() const {
  return this != internal_default_instance() && shape_ != nullptr;
}
inline bool BundleEntryProto::has_shape() const {
  return _internal_has_shape();
}
inline const ::tensorflow::TensorShapeProto& BundleEntryProto::_internal_shape() const {
  const ::tensorflow::TensorShapeProto* p = shape_;
  return p != nullptr ? *p : reinterpret_cast<const ::tensorflow::TensorShapeProto&>(
      ::tensorflow::_TensorShapeProto_default_instance_);
}
inline const ::tensorflow::TensorShapeProto& BundleEntryProto::shape() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.shape)
  return _internal_shape();
}
inline void BundleEntryProto::unsafe_arena_set_allocated_shape(
    ::tensorflow::TensorShapeProto* shape) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(shape_);
  }
  shape_ = shape;
  if (shape) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.BundleEntryProto.shape)
}
inline ::tensorflow::TensorShapeProto* BundleEntryProto::release_shape() {
  
  ::tensorflow::TensorShapeProto* temp = shape_;
  shape_ = nullptr;
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  auto* old =  reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(temp);
  temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  if (GetArenaForAllocation() == nullptr) { delete old; }
#else  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (GetArenaForAllocation() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
  return temp;
}
inline ::tensorflow::TensorShapeProto* BundleEntryProto::unsafe_arena_release_shape() {
  // @@protoc_insertion_point(field_release:tensorflow.BundleEntryProto.shape)
  
  ::tensorflow::TensorShapeProto* temp = shape_;
  shape_ = nullptr;
  return temp;
}
inline ::tensorflow::TensorShapeProto* BundleEntryProto::_internal_mutable_shape() {
  
  if (shape_ == nullptr) {
    auto* p = CreateMaybeMessage<::tensorflow::TensorShapeProto>(GetArenaForAllocation());
    shape_ = p;
  }
  return shape_;
}
inline ::tensorflow::TensorShapeProto* BundleEntryProto::mutable_shape() {
  ::tensorflow::TensorShapeProto* _msg = _internal_mutable_shape();
  // @@protoc_insertion_point(field_mutable:tensorflow.BundleEntryProto.shape)
  return _msg;
}
inline void BundleEntryProto::set_allocated_shape(::tensorflow::TensorShapeProto* shape) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(shape_);
  }
  if (shape) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<
            ::PROTOBUF_NAMESPACE_ID::MessageLite>::GetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(shape));
    if (message_arena != submessage_arena) {
      shape = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, shape, submessage_arena);
    }
    
  } else {
    
  }
  shape_ = shape;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.BundleEntryProto.shape)
}

// int32 shard_id = 3;
inline void BundleEntryProto::clear_shard_id() {
  shard_id_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BundleEntryProto::_internal_shard_id() const {
  return shard_id_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 BundleEntryProto::shard_id() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.shard_id)
  return _internal_shard_id();
}
inline void BundleEntryProto::_internal_set_shard_id(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  shard_id_ = value;
}
inline void BundleEntryProto::set_shard_id(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_shard_id(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleEntryProto.shard_id)
}

// int64 offset = 4;
inline void BundleEntryProto::clear_offset() {
  offset_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 BundleEntryProto::_internal_offset() const {
  return offset_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 BundleEntryProto::offset() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.offset)
  return _internal_offset();
}
inline void BundleEntryProto::_internal_set_offset(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  offset_ = value;
}
inline void BundleEntryProto::set_offset(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_offset(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleEntryProto.offset)
}

// int64 size = 5;
inline void BundleEntryProto::clear_size() {
  size_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 BundleEntryProto::_internal_size() const {
  return size_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 BundleEntryProto::size() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.size)
  return _internal_size();
}
inline void BundleEntryProto::_internal_set_size(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  size_ = value;
}
inline void BundleEntryProto::set_size(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_size(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleEntryProto.size)
}

// fixed32 crc32c = 6;
inline void BundleEntryProto::clear_crc32c() {
  crc32c_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 BundleEntryProto::_internal_crc32c() const {
  return crc32c_;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 BundleEntryProto::crc32c() const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.crc32c)
  return _internal_crc32c();
}
inline void BundleEntryProto::_internal_set_crc32c(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  crc32c_ = value;
}
inline void BundleEntryProto::set_crc32c(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  _internal_set_crc32c(value);
  // @@protoc_insertion_point(field_set:tensorflow.BundleEntryProto.crc32c)
}

// repeated .tensorflow.TensorSliceProto slices = 7;
inline int BundleEntryProto::_internal_slices_size() const {
  return slices_.size();
}
inline int BundleEntryProto::slices_size() const {
  return _internal_slices_size();
}
inline ::tensorflow::TensorSliceProto* BundleEntryProto::mutable_slices(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.BundleEntryProto.slices)
  return slices_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::TensorSliceProto >*
BundleEntryProto::mutable_slices() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.BundleEntryProto.slices)
  return &slices_;
}
inline const ::tensorflow::TensorSliceProto& BundleEntryProto::_internal_slices(int index) const {
  return slices_.Get(index);
}
inline const ::tensorflow::TensorSliceProto& BundleEntryProto::slices(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.BundleEntryProto.slices)
  return _internal_slices(index);
}
inline ::tensorflow::TensorSliceProto* BundleEntryProto::_internal_add_slices() {
  return slices_.Add();
}
inline ::tensorflow::TensorSliceProto* BundleEntryProto::add_slices() {
  ::tensorflow::TensorSliceProto* _add = _internal_add_slices();
  // @@protoc_insertion_point(field_add:tensorflow.BundleEntryProto.slices)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::TensorSliceProto >&
BundleEntryProto::slices() const {
  // @@protoc_insertion_point(field_list:tensorflow.BundleEntryProto.slices)
  return slices_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::tensorflow::BundleHeaderProto_Endianness> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::tensorflow::BundleHeaderProto_Endianness>() {
  return ::tensorflow::BundleHeaderProto_Endianness_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensor_5fbundle_2eproto
