// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/util/quantization/uniform_quant_ops_attr.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto;
namespace tensorflow {
class UniformQuantizedConvolutionDimensionNumbersAttr;
struct UniformQuantizedConvolutionDimensionNumbersAttrDefaultTypeInternal;
extern UniformQuantizedConvolutionDimensionNumbersAttrDefaultTypeInternal _UniformQuantizedConvolutionDimensionNumbersAttr_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr* Arena::CreateMaybeMessage<::tensorflow::UniformQuantizedConvolutionDimensionNumbersAttr>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class UniformQuantizedConvolutionDimensionNumbersAttr final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr) */ {
 public:
  inline UniformQuantizedConvolutionDimensionNumbersAttr() : UniformQuantizedConvolutionDimensionNumbersAttr(nullptr) {}
  ~UniformQuantizedConvolutionDimensionNumbersAttr() override;
  explicit constexpr UniformQuantizedConvolutionDimensionNumbersAttr(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  UniformQuantizedConvolutionDimensionNumbersAttr(const UniformQuantizedConvolutionDimensionNumbersAttr& from);
  UniformQuantizedConvolutionDimensionNumbersAttr(UniformQuantizedConvolutionDimensionNumbersAttr&& from) noexcept
    : UniformQuantizedConvolutionDimensionNumbersAttr() {
    *this = ::std::move(from);
  }

  inline UniformQuantizedConvolutionDimensionNumbersAttr& operator=(const UniformQuantizedConvolutionDimensionNumbersAttr& from) {
    CopyFrom(from);
    return *this;
  }
  inline UniformQuantizedConvolutionDimensionNumbersAttr& operator=(UniformQuantizedConvolutionDimensionNumbersAttr&& from) noexcept {
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
  static const UniformQuantizedConvolutionDimensionNumbersAttr& default_instance() {
    return *internal_default_instance();
  }
  static inline const UniformQuantizedConvolutionDimensionNumbersAttr* internal_default_instance() {
    return reinterpret_cast<const UniformQuantizedConvolutionDimensionNumbersAttr*>(
               &_UniformQuantizedConvolutionDimensionNumbersAttr_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(UniformQuantizedConvolutionDimensionNumbersAttr& a, UniformQuantizedConvolutionDimensionNumbersAttr& b) {
    a.Swap(&b);
  }
  inline void Swap(UniformQuantizedConvolutionDimensionNumbersAttr* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(UniformQuantizedConvolutionDimensionNumbersAttr* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline UniformQuantizedConvolutionDimensionNumbersAttr* New() const final {
    return new UniformQuantizedConvolutionDimensionNumbersAttr();
  }

  UniformQuantizedConvolutionDimensionNumbersAttr* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<UniformQuantizedConvolutionDimensionNumbersAttr>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const UniformQuantizedConvolutionDimensionNumbersAttr& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const UniformQuantizedConvolutionDimensionNumbersAttr& from);
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
  void InternalSwap(UniformQuantizedConvolutionDimensionNumbersAttr* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr";
  }
  protected:
  explicit UniformQuantizedConvolutionDimensionNumbersAttr(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kInputSpatialDimensionsFieldNumber = 3,
    kKernelSpatialDimensionsFieldNumber = 6,
    kOutputSpatialDimensionsFieldNumber = 9,
    kInputBatchDimensionFieldNumber = 1,
    kInputFeatureDimensionFieldNumber = 2,
    kKernelInputFeatureDimensionFieldNumber = 4,
    kKernelOutputFeatureDimensionFieldNumber = 5,
    kOutputBatchDimensionFieldNumber = 7,
    kOutputFeatureDimensionFieldNumber = 8,
  };
  // repeated int64 input_spatial_dimensions = 3;
  int input_spatial_dimensions_size() const;
  private:
  int _internal_input_spatial_dimensions_size() const;
  public:
  void clear_input_spatial_dimensions();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_input_spatial_dimensions(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_input_spatial_dimensions() const;
  void _internal_add_input_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_input_spatial_dimensions();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 input_spatial_dimensions(int index) const;
  void set_input_spatial_dimensions(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_input_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      input_spatial_dimensions() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_input_spatial_dimensions();

  // repeated int64 kernel_spatial_dimensions = 6;
  int kernel_spatial_dimensions_size() const;
  private:
  int _internal_kernel_spatial_dimensions_size() const;
  public:
  void clear_kernel_spatial_dimensions();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_kernel_spatial_dimensions(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_kernel_spatial_dimensions() const;
  void _internal_add_kernel_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_kernel_spatial_dimensions();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 kernel_spatial_dimensions(int index) const;
  void set_kernel_spatial_dimensions(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_kernel_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      kernel_spatial_dimensions() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_kernel_spatial_dimensions();

  // repeated int64 output_spatial_dimensions = 9;
  int output_spatial_dimensions_size() const;
  private:
  int _internal_output_spatial_dimensions_size() const;
  public:
  void clear_output_spatial_dimensions();
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_output_spatial_dimensions(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      _internal_output_spatial_dimensions() const;
  void _internal_add_output_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      _internal_mutable_output_spatial_dimensions();
  public:
  ::PROTOBUF_NAMESPACE_ID::int64 output_spatial_dimensions(int index) const;
  void set_output_spatial_dimensions(int index, ::PROTOBUF_NAMESPACE_ID::int64 value);
  void add_output_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
      output_spatial_dimensions() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
      mutable_output_spatial_dimensions();

  // int64 input_batch_dimension = 1;
  void clear_input_batch_dimension();
  ::PROTOBUF_NAMESPACE_ID::int64 input_batch_dimension() const;
  void set_input_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_input_batch_dimension() const;
  void _internal_set_input_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 input_feature_dimension = 2;
  void clear_input_feature_dimension();
  ::PROTOBUF_NAMESPACE_ID::int64 input_feature_dimension() const;
  void set_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_input_feature_dimension() const;
  void _internal_set_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 kernel_input_feature_dimension = 4;
  void clear_kernel_input_feature_dimension();
  ::PROTOBUF_NAMESPACE_ID::int64 kernel_input_feature_dimension() const;
  void set_kernel_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_kernel_input_feature_dimension() const;
  void _internal_set_kernel_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 kernel_output_feature_dimension = 5;
  void clear_kernel_output_feature_dimension();
  ::PROTOBUF_NAMESPACE_ID::int64 kernel_output_feature_dimension() const;
  void set_kernel_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_kernel_output_feature_dimension() const;
  void _internal_set_kernel_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 output_batch_dimension = 7;
  void clear_output_batch_dimension();
  ::PROTOBUF_NAMESPACE_ID::int64 output_batch_dimension() const;
  void set_output_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_output_batch_dimension() const;
  void _internal_set_output_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 output_feature_dimension = 8;
  void clear_output_feature_dimension();
  ::PROTOBUF_NAMESPACE_ID::int64 output_feature_dimension() const;
  void set_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_output_feature_dimension() const;
  void _internal_set_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > input_spatial_dimensions_;
  mutable std::atomic<int> _input_spatial_dimensions_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > kernel_spatial_dimensions_;
  mutable std::atomic<int> _kernel_spatial_dimensions_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 > output_spatial_dimensions_;
  mutable std::atomic<int> _output_spatial_dimensions_cached_byte_size_;
  ::PROTOBUF_NAMESPACE_ID::int64 input_batch_dimension_;
  ::PROTOBUF_NAMESPACE_ID::int64 input_feature_dimension_;
  ::PROTOBUF_NAMESPACE_ID::int64 kernel_input_feature_dimension_;
  ::PROTOBUF_NAMESPACE_ID::int64 kernel_output_feature_dimension_;
  ::PROTOBUF_NAMESPACE_ID::int64 output_batch_dimension_;
  ::PROTOBUF_NAMESPACE_ID::int64 output_feature_dimension_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// UniformQuantizedConvolutionDimensionNumbersAttr

// int64 input_batch_dimension = 1;
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_input_batch_dimension() {
  input_batch_dimension_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_input_batch_dimension() const {
  return input_batch_dimension_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::input_batch_dimension() const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_batch_dimension)
  return _internal_input_batch_dimension();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_set_input_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  input_batch_dimension_ = value;
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_input_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_input_batch_dimension(value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_batch_dimension)
}

// int64 input_feature_dimension = 2;
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_input_feature_dimension() {
  input_feature_dimension_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_input_feature_dimension() const {
  return input_feature_dimension_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::input_feature_dimension() const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_feature_dimension)
  return _internal_input_feature_dimension();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_set_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  input_feature_dimension_ = value;
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_input_feature_dimension(value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_feature_dimension)
}

// repeated int64 input_spatial_dimensions = 3;
inline int UniformQuantizedConvolutionDimensionNumbersAttr::_internal_input_spatial_dimensions_size() const {
  return input_spatial_dimensions_.size();
}
inline int UniformQuantizedConvolutionDimensionNumbersAttr::input_spatial_dimensions_size() const {
  return _internal_input_spatial_dimensions_size();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_input_spatial_dimensions() {
  input_spatial_dimensions_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_input_spatial_dimensions(int index) const {
  return input_spatial_dimensions_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::input_spatial_dimensions(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_spatial_dimensions)
  return _internal_input_spatial_dimensions(index);
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_input_spatial_dimensions(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  input_spatial_dimensions_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_spatial_dimensions)
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_add_input_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value) {
  input_spatial_dimensions_.Add(value);
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::add_input_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_input_spatial_dimensions(value);
  // @@protoc_insertion_point(field_add:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_spatial_dimensions)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
UniformQuantizedConvolutionDimensionNumbersAttr::_internal_input_spatial_dimensions() const {
  return input_spatial_dimensions_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
UniformQuantizedConvolutionDimensionNumbersAttr::input_spatial_dimensions() const {
  // @@protoc_insertion_point(field_list:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_spatial_dimensions)
  return _internal_input_spatial_dimensions();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
UniformQuantizedConvolutionDimensionNumbersAttr::_internal_mutable_input_spatial_dimensions() {
  return &input_spatial_dimensions_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
UniformQuantizedConvolutionDimensionNumbersAttr::mutable_input_spatial_dimensions() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.input_spatial_dimensions)
  return _internal_mutable_input_spatial_dimensions();
}

// int64 kernel_input_feature_dimension = 4;
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_kernel_input_feature_dimension() {
  kernel_input_feature_dimension_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_kernel_input_feature_dimension() const {
  return kernel_input_feature_dimension_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::kernel_input_feature_dimension() const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_input_feature_dimension)
  return _internal_kernel_input_feature_dimension();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_set_kernel_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  kernel_input_feature_dimension_ = value;
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_kernel_input_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_kernel_input_feature_dimension(value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_input_feature_dimension)
}

// int64 kernel_output_feature_dimension = 5;
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_kernel_output_feature_dimension() {
  kernel_output_feature_dimension_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_kernel_output_feature_dimension() const {
  return kernel_output_feature_dimension_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::kernel_output_feature_dimension() const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_output_feature_dimension)
  return _internal_kernel_output_feature_dimension();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_set_kernel_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  kernel_output_feature_dimension_ = value;
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_kernel_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_kernel_output_feature_dimension(value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_output_feature_dimension)
}

// repeated int64 kernel_spatial_dimensions = 6;
inline int UniformQuantizedConvolutionDimensionNumbersAttr::_internal_kernel_spatial_dimensions_size() const {
  return kernel_spatial_dimensions_.size();
}
inline int UniformQuantizedConvolutionDimensionNumbersAttr::kernel_spatial_dimensions_size() const {
  return _internal_kernel_spatial_dimensions_size();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_kernel_spatial_dimensions() {
  kernel_spatial_dimensions_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_kernel_spatial_dimensions(int index) const {
  return kernel_spatial_dimensions_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::kernel_spatial_dimensions(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_spatial_dimensions)
  return _internal_kernel_spatial_dimensions(index);
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_kernel_spatial_dimensions(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  kernel_spatial_dimensions_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_spatial_dimensions)
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_add_kernel_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value) {
  kernel_spatial_dimensions_.Add(value);
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::add_kernel_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_kernel_spatial_dimensions(value);
  // @@protoc_insertion_point(field_add:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_spatial_dimensions)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
UniformQuantizedConvolutionDimensionNumbersAttr::_internal_kernel_spatial_dimensions() const {
  return kernel_spatial_dimensions_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
UniformQuantizedConvolutionDimensionNumbersAttr::kernel_spatial_dimensions() const {
  // @@protoc_insertion_point(field_list:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_spatial_dimensions)
  return _internal_kernel_spatial_dimensions();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
UniformQuantizedConvolutionDimensionNumbersAttr::_internal_mutable_kernel_spatial_dimensions() {
  return &kernel_spatial_dimensions_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
UniformQuantizedConvolutionDimensionNumbersAttr::mutable_kernel_spatial_dimensions() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.kernel_spatial_dimensions)
  return _internal_mutable_kernel_spatial_dimensions();
}

// int64 output_batch_dimension = 7;
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_output_batch_dimension() {
  output_batch_dimension_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_output_batch_dimension() const {
  return output_batch_dimension_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::output_batch_dimension() const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_batch_dimension)
  return _internal_output_batch_dimension();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_set_output_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  output_batch_dimension_ = value;
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_output_batch_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_output_batch_dimension(value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_batch_dimension)
}

// int64 output_feature_dimension = 8;
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_output_feature_dimension() {
  output_feature_dimension_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_output_feature_dimension() const {
  return output_feature_dimension_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::output_feature_dimension() const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_feature_dimension)
  return _internal_output_feature_dimension();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_set_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  output_feature_dimension_ = value;
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_output_feature_dimension(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_output_feature_dimension(value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_feature_dimension)
}

// repeated int64 output_spatial_dimensions = 9;
inline int UniformQuantizedConvolutionDimensionNumbersAttr::_internal_output_spatial_dimensions_size() const {
  return output_spatial_dimensions_.size();
}
inline int UniformQuantizedConvolutionDimensionNumbersAttr::output_spatial_dimensions_size() const {
  return _internal_output_spatial_dimensions_size();
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::clear_output_spatial_dimensions() {
  output_spatial_dimensions_.Clear();
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::_internal_output_spatial_dimensions(int index) const {
  return output_spatial_dimensions_.Get(index);
}
inline ::PROTOBUF_NAMESPACE_ID::int64 UniformQuantizedConvolutionDimensionNumbersAttr::output_spatial_dimensions(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_spatial_dimensions)
  return _internal_output_spatial_dimensions(index);
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::set_output_spatial_dimensions(int index, ::PROTOBUF_NAMESPACE_ID::int64 value) {
  output_spatial_dimensions_.Set(index, value);
  // @@protoc_insertion_point(field_set:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_spatial_dimensions)
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::_internal_add_output_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value) {
  output_spatial_dimensions_.Add(value);
}
inline void UniformQuantizedConvolutionDimensionNumbersAttr::add_output_spatial_dimensions(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_add_output_spatial_dimensions(value);
  // @@protoc_insertion_point(field_add:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_spatial_dimensions)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
UniformQuantizedConvolutionDimensionNumbersAttr::_internal_output_spatial_dimensions() const {
  return output_spatial_dimensions_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >&
UniformQuantizedConvolutionDimensionNumbersAttr::output_spatial_dimensions() const {
  // @@protoc_insertion_point(field_list:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_spatial_dimensions)
  return _internal_output_spatial_dimensions();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
UniformQuantizedConvolutionDimensionNumbersAttr::_internal_mutable_output_spatial_dimensions() {
  return &output_spatial_dimensions_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< ::PROTOBUF_NAMESPACE_ID::int64 >*
UniformQuantizedConvolutionDimensionNumbersAttr::mutable_output_spatial_dimensions() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.UniformQuantizedConvolutionDimensionNumbersAttr.output_spatial_dimensions)
  return _internal_mutable_output_spatial_dimensions();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2futil_2fquantization_2funiform_5fquant_5fops_5fattr_2eproto
