// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/tsl/protobuf/histogram.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[1]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto;
namespace tensorflow {
class HistogramProto;
struct HistogramProtoDefaultTypeInternal;
extern HistogramProtoDefaultTypeInternal _HistogramProto_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::HistogramProto* Arena::CreateMaybeMessage<::tensorflow::HistogramProto>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class HistogramProto final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.HistogramProto) */ {
 public:
  inline HistogramProto() : HistogramProto(nullptr) {}
  ~HistogramProto() override;
  explicit constexpr HistogramProto(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  HistogramProto(const HistogramProto& from);
  HistogramProto(HistogramProto&& from) noexcept
    : HistogramProto() {
    *this = ::std::move(from);
  }

  inline HistogramProto& operator=(const HistogramProto& from) {
    CopyFrom(from);
    return *this;
  }
  inline HistogramProto& operator=(HistogramProto&& from) noexcept {
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
  static const HistogramProto& default_instance() {
    return *internal_default_instance();
  }
  static inline const HistogramProto* internal_default_instance() {
    return reinterpret_cast<const HistogramProto*>(
               &_HistogramProto_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(HistogramProto& a, HistogramProto& b) {
    a.Swap(&b);
  }
  inline void Swap(HistogramProto* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(HistogramProto* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline HistogramProto* New() const final {
    return new HistogramProto();
  }

  HistogramProto* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<HistogramProto>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const HistogramProto& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const HistogramProto& from);
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
  void InternalSwap(HistogramProto* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.HistogramProto";
  }
  protected:
  explicit HistogramProto(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kBucketLimitFieldNumber = 6,
    kBucketFieldNumber = 7,
    kMinFieldNumber = 1,
    kMaxFieldNumber = 2,
    kNumFieldNumber = 3,
    kSumFieldNumber = 4,
    kSumSquaresFieldNumber = 5,
  };
  // repeated double bucket_limit = 6 [packed = true];
  int bucket_limit_size() const;
  private:
  int _internal_bucket_limit_size() const;
  public:
  void clear_bucket_limit();
  private:
  double _internal_bucket_limit(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      _internal_bucket_limit() const;
  void _internal_add_bucket_limit(double value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      _internal_mutable_bucket_limit();
  public:
  double bucket_limit(int index) const;
  void set_bucket_limit(int index, double value);
  void add_bucket_limit(double value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      bucket_limit() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      mutable_bucket_limit();

  // repeated double bucket = 7 [packed = true];
  int bucket_size() const;
  private:
  int _internal_bucket_size() const;
  public:
  void clear_bucket();
  private:
  double _internal_bucket(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      _internal_bucket() const;
  void _internal_add_bucket(double value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      _internal_mutable_bucket();
  public:
  double bucket(int index) const;
  void set_bucket(int index, double value);
  void add_bucket(double value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
      bucket() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
      mutable_bucket();

  // double min = 1;
  void clear_min();
  double min() const;
  void set_min(double value);
  private:
  double _internal_min() const;
  void _internal_set_min(double value);
  public:

  // double max = 2;
  void clear_max();
  double max() const;
  void set_max(double value);
  private:
  double _internal_max() const;
  void _internal_set_max(double value);
  public:

  // double num = 3;
  void clear_num();
  double num() const;
  void set_num(double value);
  private:
  double _internal_num() const;
  void _internal_set_num(double value);
  public:

  // double sum = 4;
  void clear_sum();
  double sum() const;
  void set_sum(double value);
  private:
  double _internal_sum() const;
  void _internal_set_sum(double value);
  public:

  // double sum_squares = 5;
  void clear_sum_squares();
  double sum_squares() const;
  void set_sum_squares(double value);
  private:
  double _internal_sum_squares() const;
  void _internal_set_sum_squares(double value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.HistogramProto)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double > bucket_limit_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< double > bucket_;
  double min_;
  double max_;
  double num_;
  double sum_;
  double sum_squares_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// HistogramProto

// double min = 1;
inline void HistogramProto::clear_min() {
  min_ = 0;
}
inline double HistogramProto::_internal_min() const {
  return min_;
}
inline double HistogramProto::min() const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.min)
  return _internal_min();
}
inline void HistogramProto::_internal_set_min(double value) {
  
  min_ = value;
}
inline void HistogramProto::set_min(double value) {
  _internal_set_min(value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.min)
}

// double max = 2;
inline void HistogramProto::clear_max() {
  max_ = 0;
}
inline double HistogramProto::_internal_max() const {
  return max_;
}
inline double HistogramProto::max() const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.max)
  return _internal_max();
}
inline void HistogramProto::_internal_set_max(double value) {
  
  max_ = value;
}
inline void HistogramProto::set_max(double value) {
  _internal_set_max(value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.max)
}

// double num = 3;
inline void HistogramProto::clear_num() {
  num_ = 0;
}
inline double HistogramProto::_internal_num() const {
  return num_;
}
inline double HistogramProto::num() const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.num)
  return _internal_num();
}
inline void HistogramProto::_internal_set_num(double value) {
  
  num_ = value;
}
inline void HistogramProto::set_num(double value) {
  _internal_set_num(value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.num)
}

// double sum = 4;
inline void HistogramProto::clear_sum() {
  sum_ = 0;
}
inline double HistogramProto::_internal_sum() const {
  return sum_;
}
inline double HistogramProto::sum() const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.sum)
  return _internal_sum();
}
inline void HistogramProto::_internal_set_sum(double value) {
  
  sum_ = value;
}
inline void HistogramProto::set_sum(double value) {
  _internal_set_sum(value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.sum)
}

// double sum_squares = 5;
inline void HistogramProto::clear_sum_squares() {
  sum_squares_ = 0;
}
inline double HistogramProto::_internal_sum_squares() const {
  return sum_squares_;
}
inline double HistogramProto::sum_squares() const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.sum_squares)
  return _internal_sum_squares();
}
inline void HistogramProto::_internal_set_sum_squares(double value) {
  
  sum_squares_ = value;
}
inline void HistogramProto::set_sum_squares(double value) {
  _internal_set_sum_squares(value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.sum_squares)
}

// repeated double bucket_limit = 6 [packed = true];
inline int HistogramProto::_internal_bucket_limit_size() const {
  return bucket_limit_.size();
}
inline int HistogramProto::bucket_limit_size() const {
  return _internal_bucket_limit_size();
}
inline void HistogramProto::clear_bucket_limit() {
  bucket_limit_.Clear();
}
inline double HistogramProto::_internal_bucket_limit(int index) const {
  return bucket_limit_.Get(index);
}
inline double HistogramProto::bucket_limit(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.bucket_limit)
  return _internal_bucket_limit(index);
}
inline void HistogramProto::set_bucket_limit(int index, double value) {
  bucket_limit_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.bucket_limit)
}
inline void HistogramProto::_internal_add_bucket_limit(double value) {
  bucket_limit_.Add(value);
}
inline void HistogramProto::add_bucket_limit(double value) {
  _internal_add_bucket_limit(value);
  // @@protoc_insertion_point(field_add:tensorflow.HistogramProto.bucket_limit)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
HistogramProto::_internal_bucket_limit() const {
  return bucket_limit_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
HistogramProto::bucket_limit() const {
  // @@protoc_insertion_point(field_list:tensorflow.HistogramProto.bucket_limit)
  return _internal_bucket_limit();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
HistogramProto::_internal_mutable_bucket_limit() {
  return &bucket_limit_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
HistogramProto::mutable_bucket_limit() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.HistogramProto.bucket_limit)
  return _internal_mutable_bucket_limit();
}

// repeated double bucket = 7 [packed = true];
inline int HistogramProto::_internal_bucket_size() const {
  return bucket_.size();
}
inline int HistogramProto::bucket_size() const {
  return _internal_bucket_size();
}
inline void HistogramProto::clear_bucket() {
  bucket_.Clear();
}
inline double HistogramProto::_internal_bucket(int index) const {
  return bucket_.Get(index);
}
inline double HistogramProto::bucket(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.HistogramProto.bucket)
  return _internal_bucket(index);
}
inline void HistogramProto::set_bucket(int index, double value) {
  bucket_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.HistogramProto.bucket)
}
inline void HistogramProto::_internal_add_bucket(double value) {
  bucket_.Add(value);
}
inline void HistogramProto::add_bucket(double value) {
  _internal_add_bucket(value);
  // @@protoc_insertion_point(field_add:tensorflow.HistogramProto.bucket)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
HistogramProto::_internal_bucket() const {
  return bucket_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >&
HistogramProto::bucket() const {
  // @@protoc_insertion_point(field_list:tensorflow.HistogramProto.bucket)
  return _internal_bucket();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
HistogramProto::_internal_mutable_bucket() {
  return &bucket_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< double >*
HistogramProto::mutable_bucket() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.HistogramProto.bucket)
  return _internal_mutable_bucket();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2ftsl_2fprotobuf_2fhistogram_2eproto
