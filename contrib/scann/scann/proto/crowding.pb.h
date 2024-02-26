// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: scann/proto/crowding.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_scann_2fproto_2fcrowding_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_scann_2fproto_2fcrowding_2eproto

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
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_scann_2fproto_2fcrowding_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_scann_2fproto_2fcrowding_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_scann_2fproto_2fcrowding_2eproto;
namespace research_scann {
class Crowding;
struct CrowdingDefaultTypeInternal;
extern CrowdingDefaultTypeInternal _Crowding_default_instance_;
class Crowding_Offline;
struct Crowding_OfflineDefaultTypeInternal;
extern Crowding_OfflineDefaultTypeInternal _Crowding_Offline_default_instance_;
}  // namespace research_scann
PROTOBUF_NAMESPACE_OPEN
template<> ::research_scann::Crowding* Arena::CreateMaybeMessage<::research_scann::Crowding>(Arena*);
template<> ::research_scann::Crowding_Offline* Arena::CreateMaybeMessage<::research_scann::Crowding_Offline>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace research_scann {

// ===================================================================

class Crowding_Offline final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:research_scann.Crowding.Offline) */ {
 public:
  inline Crowding_Offline() : Crowding_Offline(nullptr) {}
  ~Crowding_Offline() override;
  explicit constexpr Crowding_Offline(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Crowding_Offline(const Crowding_Offline& from);
  Crowding_Offline(Crowding_Offline&& from) noexcept
    : Crowding_Offline() {
    *this = ::std::move(from);
  }

  inline Crowding_Offline& operator=(const Crowding_Offline& from) {
    CopyFrom(from);
    return *this;
  }
  inline Crowding_Offline& operator=(Crowding_Offline&& from) noexcept {
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
  static const Crowding_Offline& default_instance() {
    return *internal_default_instance();
  }
  static inline const Crowding_Offline* internal_default_instance() {
    return reinterpret_cast<const Crowding_Offline*>(
               &_Crowding_Offline_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(Crowding_Offline& a, Crowding_Offline& b) {
    a.Swap(&b);
  }
  inline void Swap(Crowding_Offline* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Crowding_Offline* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Crowding_Offline* New() const final {
    return new Crowding_Offline();
  }

  Crowding_Offline* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Crowding_Offline>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const Crowding_Offline& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const Crowding_Offline& from);
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
  void InternalSwap(Crowding_Offline* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "research_scann.Crowding.Offline";
  }
  protected:
  explicit Crowding_Offline(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kPerCrowdingAttributePreReorderingNumNeighborsFieldNumber = 1,
    kPerCrowdingAttributePostReorderingNumNeighborsFieldNumber = 2,
  };
  // optional int32 per_crowding_attribute_pre_reordering_num_neighbors = 1 [default = 2147483647];
  bool has_per_crowding_attribute_pre_reordering_num_neighbors() const;
  private:
  bool _internal_has_per_crowding_attribute_pre_reordering_num_neighbors() const;
  public:
  void clear_per_crowding_attribute_pre_reordering_num_neighbors();
  ::PROTOBUF_NAMESPACE_ID::int32 per_crowding_attribute_pre_reordering_num_neighbors() const;
  void set_per_crowding_attribute_pre_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_per_crowding_attribute_pre_reordering_num_neighbors() const;
  void _internal_set_per_crowding_attribute_pre_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // optional int32 per_crowding_attribute_post_reordering_num_neighbors = 2 [default = 2147483647];
  bool has_per_crowding_attribute_post_reordering_num_neighbors() const;
  private:
  bool _internal_has_per_crowding_attribute_post_reordering_num_neighbors() const;
  public:
  void clear_per_crowding_attribute_post_reordering_num_neighbors();
  ::PROTOBUF_NAMESPACE_ID::int32 per_crowding_attribute_post_reordering_num_neighbors() const;
  void set_per_crowding_attribute_post_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_per_crowding_attribute_post_reordering_num_neighbors() const;
  void _internal_set_per_crowding_attribute_post_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:research_scann.Crowding.Offline)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::int32 per_crowding_attribute_pre_reordering_num_neighbors_;
  ::PROTOBUF_NAMESPACE_ID::int32 per_crowding_attribute_post_reordering_num_neighbors_;
  friend struct ::TableStruct_scann_2fproto_2fcrowding_2eproto;
};
// -------------------------------------------------------------------

class Crowding final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:research_scann.Crowding) */ {
 public:
  inline Crowding() : Crowding(nullptr) {}
  ~Crowding() override;
  explicit constexpr Crowding(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  Crowding(const Crowding& from);
  Crowding(Crowding&& from) noexcept
    : Crowding() {
    *this = ::std::move(from);
  }

  inline Crowding& operator=(const Crowding& from) {
    CopyFrom(from);
    return *this;
  }
  inline Crowding& operator=(Crowding&& from) noexcept {
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
  static const Crowding& default_instance() {
    return *internal_default_instance();
  }
  static inline const Crowding* internal_default_instance() {
    return reinterpret_cast<const Crowding*>(
               &_Crowding_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(Crowding& a, Crowding& b) {
    a.Swap(&b);
  }
  inline void Swap(Crowding* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(Crowding* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Crowding* New() const final {
    return new Crowding();
  }

  Crowding* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Crowding>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const Crowding& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const Crowding& from);
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
  void InternalSwap(Crowding* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "research_scann.Crowding";
  }
  protected:
  explicit Crowding(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef Crowding_Offline Offline;

  // accessors -------------------------------------------------------

  enum : int {
    kOfflineFieldNumber = 2,
    kEnabledFieldNumber = 1,
  };
  // optional .research_scann.Crowding.Offline offline = 2;
  bool has_offline() const;
  private:
  bool _internal_has_offline() const;
  public:
  void clear_offline();
  const ::research_scann::Crowding_Offline& offline() const;
  PROTOBUF_MUST_USE_RESULT ::research_scann::Crowding_Offline* release_offline();
  ::research_scann::Crowding_Offline* mutable_offline();
  void set_allocated_offline(::research_scann::Crowding_Offline* offline);
  private:
  const ::research_scann::Crowding_Offline& _internal_offline() const;
  ::research_scann::Crowding_Offline* _internal_mutable_offline();
  public:
  void unsafe_arena_set_allocated_offline(
      ::research_scann::Crowding_Offline* offline);
  ::research_scann::Crowding_Offline* unsafe_arena_release_offline();

  // optional bool enabled = 1 [default = false];
  bool has_enabled() const;
  private:
  bool _internal_has_enabled() const;
  public:
  void clear_enabled();
  bool enabled() const;
  void set_enabled(bool value);
  private:
  bool _internal_enabled() const;
  void _internal_set_enabled(bool value);
  public:

  // @@protoc_insertion_point(class_scope:research_scann.Crowding)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::research_scann::Crowding_Offline* offline_;
  bool enabled_;
  friend struct ::TableStruct_scann_2fproto_2fcrowding_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Crowding_Offline

// optional int32 per_crowding_attribute_pre_reordering_num_neighbors = 1 [default = 2147483647];
inline bool Crowding_Offline::_internal_has_per_crowding_attribute_pre_reordering_num_neighbors() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  return value;
}
inline bool Crowding_Offline::has_per_crowding_attribute_pre_reordering_num_neighbors() const {
  return _internal_has_per_crowding_attribute_pre_reordering_num_neighbors();
}
inline void Crowding_Offline::clear_per_crowding_attribute_pre_reordering_num_neighbors() {
  per_crowding_attribute_pre_reordering_num_neighbors_ = 2147483647;
  _has_bits_[0] &= ~0x00000001u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Crowding_Offline::_internal_per_crowding_attribute_pre_reordering_num_neighbors() const {
  return per_crowding_attribute_pre_reordering_num_neighbors_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Crowding_Offline::per_crowding_attribute_pre_reordering_num_neighbors() const {
  // @@protoc_insertion_point(field_get:research_scann.Crowding.Offline.per_crowding_attribute_pre_reordering_num_neighbors)
  return _internal_per_crowding_attribute_pre_reordering_num_neighbors();
}
inline void Crowding_Offline::_internal_set_per_crowding_attribute_pre_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000001u;
  per_crowding_attribute_pre_reordering_num_neighbors_ = value;
}
inline void Crowding_Offline::set_per_crowding_attribute_pre_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_per_crowding_attribute_pre_reordering_num_neighbors(value);
  // @@protoc_insertion_point(field_set:research_scann.Crowding.Offline.per_crowding_attribute_pre_reordering_num_neighbors)
}

// optional int32 per_crowding_attribute_post_reordering_num_neighbors = 2 [default = 2147483647];
inline bool Crowding_Offline::_internal_has_per_crowding_attribute_post_reordering_num_neighbors() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Crowding_Offline::has_per_crowding_attribute_post_reordering_num_neighbors() const {
  return _internal_has_per_crowding_attribute_post_reordering_num_neighbors();
}
inline void Crowding_Offline::clear_per_crowding_attribute_post_reordering_num_neighbors() {
  per_crowding_attribute_post_reordering_num_neighbors_ = 2147483647;
  _has_bits_[0] &= ~0x00000002u;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Crowding_Offline::_internal_per_crowding_attribute_post_reordering_num_neighbors() const {
  return per_crowding_attribute_post_reordering_num_neighbors_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Crowding_Offline::per_crowding_attribute_post_reordering_num_neighbors() const {
  // @@protoc_insertion_point(field_get:research_scann.Crowding.Offline.per_crowding_attribute_post_reordering_num_neighbors)
  return _internal_per_crowding_attribute_post_reordering_num_neighbors();
}
inline void Crowding_Offline::_internal_set_per_crowding_attribute_post_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _has_bits_[0] |= 0x00000002u;
  per_crowding_attribute_post_reordering_num_neighbors_ = value;
}
inline void Crowding_Offline::set_per_crowding_attribute_post_reordering_num_neighbors(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_per_crowding_attribute_post_reordering_num_neighbors(value);
  // @@protoc_insertion_point(field_set:research_scann.Crowding.Offline.per_crowding_attribute_post_reordering_num_neighbors)
}

// -------------------------------------------------------------------

// Crowding

// optional bool enabled = 1 [default = false];
inline bool Crowding::_internal_has_enabled() const {
  bool value = (_has_bits_[0] & 0x00000002u) != 0;
  return value;
}
inline bool Crowding::has_enabled() const {
  return _internal_has_enabled();
}
inline void Crowding::clear_enabled() {
  enabled_ = false;
  _has_bits_[0] &= ~0x00000002u;
}
inline bool Crowding::_internal_enabled() const {
  return enabled_;
}
inline bool Crowding::enabled() const {
  // @@protoc_insertion_point(field_get:research_scann.Crowding.enabled)
  return _internal_enabled();
}
inline void Crowding::_internal_set_enabled(bool value) {
  _has_bits_[0] |= 0x00000002u;
  enabled_ = value;
}
inline void Crowding::set_enabled(bool value) {
  _internal_set_enabled(value);
  // @@protoc_insertion_point(field_set:research_scann.Crowding.enabled)
}

// optional .research_scann.Crowding.Offline offline = 2;
inline bool Crowding::_internal_has_offline() const {
  bool value = (_has_bits_[0] & 0x00000001u) != 0;
  PROTOBUF_ASSUME(!value || offline_ != nullptr);
  return value;
}
inline bool Crowding::has_offline() const {
  return _internal_has_offline();
}
inline void Crowding::clear_offline() {
  if (offline_ != nullptr) offline_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
inline const ::research_scann::Crowding_Offline& Crowding::_internal_offline() const {
  const ::research_scann::Crowding_Offline* p = offline_;
  return p != nullptr ? *p : reinterpret_cast<const ::research_scann::Crowding_Offline&>(
      ::research_scann::_Crowding_Offline_default_instance_);
}
inline const ::research_scann::Crowding_Offline& Crowding::offline() const {
  // @@protoc_insertion_point(field_get:research_scann.Crowding.offline)
  return _internal_offline();
}
inline void Crowding::unsafe_arena_set_allocated_offline(
    ::research_scann::Crowding_Offline* offline) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(offline_);
  }
  offline_ = offline;
  if (offline) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:research_scann.Crowding.offline)
}
inline ::research_scann::Crowding_Offline* Crowding::release_offline() {
  _has_bits_[0] &= ~0x00000001u;
  ::research_scann::Crowding_Offline* temp = offline_;
  offline_ = nullptr;
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
inline ::research_scann::Crowding_Offline* Crowding::unsafe_arena_release_offline() {
  // @@protoc_insertion_point(field_release:research_scann.Crowding.offline)
  _has_bits_[0] &= ~0x00000001u;
  ::research_scann::Crowding_Offline* temp = offline_;
  offline_ = nullptr;
  return temp;
}
inline ::research_scann::Crowding_Offline* Crowding::_internal_mutable_offline() {
  _has_bits_[0] |= 0x00000001u;
  if (offline_ == nullptr) {
    auto* p = CreateMaybeMessage<::research_scann::Crowding_Offline>(GetArenaForAllocation());
    offline_ = p;
  }
  return offline_;
}
inline ::research_scann::Crowding_Offline* Crowding::mutable_offline() {
  ::research_scann::Crowding_Offline* _msg = _internal_mutable_offline();
  // @@protoc_insertion_point(field_mutable:research_scann.Crowding.offline)
  return _msg;
}
inline void Crowding::set_allocated_offline(::research_scann::Crowding_Offline* offline) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete offline_;
  }
  if (offline) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<::research_scann::Crowding_Offline>::GetOwningArena(offline);
    if (message_arena != submessage_arena) {
      offline = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, offline, submessage_arena);
    }
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  offline_ = offline;
  // @@protoc_insertion_point(field_set_allocated:research_scann.Crowding.offline)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace research_scann

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_scann_2fproto_2fcrowding_2eproto
