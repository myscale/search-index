// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/graph_debug_info.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto

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
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxiliaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[4]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto;
namespace tensorflow {
class GraphDebugInfo;
struct GraphDebugInfoDefaultTypeInternal;
extern GraphDebugInfoDefaultTypeInternal _GraphDebugInfo_default_instance_;
class GraphDebugInfo_FileLineCol;
struct GraphDebugInfo_FileLineColDefaultTypeInternal;
extern GraphDebugInfo_FileLineColDefaultTypeInternal _GraphDebugInfo_FileLineCol_default_instance_;
class GraphDebugInfo_StackTrace;
struct GraphDebugInfo_StackTraceDefaultTypeInternal;
extern GraphDebugInfo_StackTraceDefaultTypeInternal _GraphDebugInfo_StackTrace_default_instance_;
class GraphDebugInfo_TracesEntry_DoNotUse;
struct GraphDebugInfo_TracesEntry_DoNotUseDefaultTypeInternal;
extern GraphDebugInfo_TracesEntry_DoNotUseDefaultTypeInternal _GraphDebugInfo_TracesEntry_DoNotUse_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::GraphDebugInfo* Arena::CreateMaybeMessage<::tensorflow::GraphDebugInfo>(Arena*);
template<> ::tensorflow::GraphDebugInfo_FileLineCol* Arena::CreateMaybeMessage<::tensorflow::GraphDebugInfo_FileLineCol>(Arena*);
template<> ::tensorflow::GraphDebugInfo_StackTrace* Arena::CreateMaybeMessage<::tensorflow::GraphDebugInfo_StackTrace>(Arena*);
template<> ::tensorflow::GraphDebugInfo_TracesEntry_DoNotUse* Arena::CreateMaybeMessage<::tensorflow::GraphDebugInfo_TracesEntry_DoNotUse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class GraphDebugInfo_FileLineCol final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.GraphDebugInfo.FileLineCol) */ {
 public:
  inline GraphDebugInfo_FileLineCol() : GraphDebugInfo_FileLineCol(nullptr) {}
  ~GraphDebugInfo_FileLineCol() override;
  explicit constexpr GraphDebugInfo_FileLineCol(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GraphDebugInfo_FileLineCol(const GraphDebugInfo_FileLineCol& from);
  GraphDebugInfo_FileLineCol(GraphDebugInfo_FileLineCol&& from) noexcept
    : GraphDebugInfo_FileLineCol() {
    *this = ::std::move(from);
  }

  inline GraphDebugInfo_FileLineCol& operator=(const GraphDebugInfo_FileLineCol& from) {
    CopyFrom(from);
    return *this;
  }
  inline GraphDebugInfo_FileLineCol& operator=(GraphDebugInfo_FileLineCol&& from) noexcept {
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
  static const GraphDebugInfo_FileLineCol& default_instance() {
    return *internal_default_instance();
  }
  static inline const GraphDebugInfo_FileLineCol* internal_default_instance() {
    return reinterpret_cast<const GraphDebugInfo_FileLineCol*>(
               &_GraphDebugInfo_FileLineCol_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(GraphDebugInfo_FileLineCol& a, GraphDebugInfo_FileLineCol& b) {
    a.Swap(&b);
  }
  inline void Swap(GraphDebugInfo_FileLineCol* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(GraphDebugInfo_FileLineCol* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GraphDebugInfo_FileLineCol* New() const final {
    return new GraphDebugInfo_FileLineCol();
  }

  GraphDebugInfo_FileLineCol* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GraphDebugInfo_FileLineCol>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const GraphDebugInfo_FileLineCol& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const GraphDebugInfo_FileLineCol& from);
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
  void InternalSwap(GraphDebugInfo_FileLineCol* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.GraphDebugInfo.FileLineCol";
  }
  protected:
  explicit GraphDebugInfo_FileLineCol(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kFuncFieldNumber = 4,
    kCodeFieldNumber = 5,
    kFileIndexFieldNumber = 1,
    kLineFieldNumber = 2,
    kColFieldNumber = 3,
  };
  // string func = 4;
  void clear_func();
  const std::string& func() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_func(ArgT0&& arg0, ArgT... args);
  std::string* mutable_func();
  PROTOBUF_MUST_USE_RESULT std::string* release_func();
  void set_allocated_func(std::string* func);
  private:
  const std::string& _internal_func() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_func(const std::string& value);
  std::string* _internal_mutable_func();
  public:

  // string code = 5;
  void clear_code();
  const std::string& code() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_code(ArgT0&& arg0, ArgT... args);
  std::string* mutable_code();
  PROTOBUF_MUST_USE_RESULT std::string* release_code();
  void set_allocated_code(std::string* code);
  private:
  const std::string& _internal_code() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_code(const std::string& value);
  std::string* _internal_mutable_code();
  public:

  // int32 file_index = 1;
  void clear_file_index();
  ::PROTOBUF_NAMESPACE_ID::int32 file_index() const;
  void set_file_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_file_index() const;
  void _internal_set_file_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 line = 2;
  void clear_line();
  ::PROTOBUF_NAMESPACE_ID::int32 line() const;
  void set_line(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_line() const;
  void _internal_set_line(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 col = 3;
  void clear_col();
  ::PROTOBUF_NAMESPACE_ID::int32 col() const;
  void set_col(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_col() const;
  void _internal_set_col(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.GraphDebugInfo.FileLineCol)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr func_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr code_;
  ::PROTOBUF_NAMESPACE_ID::int32 file_index_;
  ::PROTOBUF_NAMESPACE_ID::int32 line_;
  ::PROTOBUF_NAMESPACE_ID::int32 col_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto;
};
// -------------------------------------------------------------------

class GraphDebugInfo_StackTrace final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.GraphDebugInfo.StackTrace) */ {
 public:
  inline GraphDebugInfo_StackTrace() : GraphDebugInfo_StackTrace(nullptr) {}
  ~GraphDebugInfo_StackTrace() override;
  explicit constexpr GraphDebugInfo_StackTrace(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GraphDebugInfo_StackTrace(const GraphDebugInfo_StackTrace& from);
  GraphDebugInfo_StackTrace(GraphDebugInfo_StackTrace&& from) noexcept
    : GraphDebugInfo_StackTrace() {
    *this = ::std::move(from);
  }

  inline GraphDebugInfo_StackTrace& operator=(const GraphDebugInfo_StackTrace& from) {
    CopyFrom(from);
    return *this;
  }
  inline GraphDebugInfo_StackTrace& operator=(GraphDebugInfo_StackTrace&& from) noexcept {
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
  static const GraphDebugInfo_StackTrace& default_instance() {
    return *internal_default_instance();
  }
  static inline const GraphDebugInfo_StackTrace* internal_default_instance() {
    return reinterpret_cast<const GraphDebugInfo_StackTrace*>(
               &_GraphDebugInfo_StackTrace_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(GraphDebugInfo_StackTrace& a, GraphDebugInfo_StackTrace& b) {
    a.Swap(&b);
  }
  inline void Swap(GraphDebugInfo_StackTrace* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(GraphDebugInfo_StackTrace* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GraphDebugInfo_StackTrace* New() const final {
    return new GraphDebugInfo_StackTrace();
  }

  GraphDebugInfo_StackTrace* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GraphDebugInfo_StackTrace>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const GraphDebugInfo_StackTrace& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const GraphDebugInfo_StackTrace& from);
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
  void InternalSwap(GraphDebugInfo_StackTrace* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.GraphDebugInfo.StackTrace";
  }
  protected:
  explicit GraphDebugInfo_StackTrace(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kFileLineColsFieldNumber = 1,
  };
  // repeated .tensorflow.GraphDebugInfo.FileLineCol file_line_cols = 1;
  int file_line_cols_size() const;
  private:
  int _internal_file_line_cols_size() const;
  public:
  void clear_file_line_cols();
  ::tensorflow::GraphDebugInfo_FileLineCol* mutable_file_line_cols(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::GraphDebugInfo_FileLineCol >*
      mutable_file_line_cols();
  private:
  const ::tensorflow::GraphDebugInfo_FileLineCol& _internal_file_line_cols(int index) const;
  ::tensorflow::GraphDebugInfo_FileLineCol* _internal_add_file_line_cols();
  public:
  const ::tensorflow::GraphDebugInfo_FileLineCol& file_line_cols(int index) const;
  ::tensorflow::GraphDebugInfo_FileLineCol* add_file_line_cols();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::GraphDebugInfo_FileLineCol >&
      file_line_cols() const;

  // @@protoc_insertion_point(class_scope:tensorflow.GraphDebugInfo.StackTrace)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::GraphDebugInfo_FileLineCol > file_line_cols_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto;
};
// -------------------------------------------------------------------

class GraphDebugInfo_TracesEntry_DoNotUse : public ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<GraphDebugInfo_TracesEntry_DoNotUse, 
    std::string, ::tensorflow::GraphDebugInfo_StackTrace,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> {
public:
  typedef ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<GraphDebugInfo_TracesEntry_DoNotUse, 
    std::string, ::tensorflow::GraphDebugInfo_StackTrace,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> SuperType;
  GraphDebugInfo_TracesEntry_DoNotUse();
  explicit constexpr GraphDebugInfo_TracesEntry_DoNotUse(
      ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);
  explicit GraphDebugInfo_TracesEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  void MergeFrom(const GraphDebugInfo_TracesEntry_DoNotUse& other);
  static const GraphDebugInfo_TracesEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const GraphDebugInfo_TracesEntry_DoNotUse*>(&_GraphDebugInfo_TracesEntry_DoNotUse_default_instance_); }
  static bool ValidateKey(std::string* s) {
    return ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(s->data(), static_cast<int>(s->size()), ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE, "tensorflow.GraphDebugInfo.TracesEntry.key");
 }
  static bool ValidateValue(void*) { return true; }
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
};

// -------------------------------------------------------------------

class GraphDebugInfo final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.GraphDebugInfo) */ {
 public:
  inline GraphDebugInfo() : GraphDebugInfo(nullptr) {}
  ~GraphDebugInfo() override;
  explicit constexpr GraphDebugInfo(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  GraphDebugInfo(const GraphDebugInfo& from);
  GraphDebugInfo(GraphDebugInfo&& from) noexcept
    : GraphDebugInfo() {
    *this = ::std::move(from);
  }

  inline GraphDebugInfo& operator=(const GraphDebugInfo& from) {
    CopyFrom(from);
    return *this;
  }
  inline GraphDebugInfo& operator=(GraphDebugInfo&& from) noexcept {
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
  static const GraphDebugInfo& default_instance() {
    return *internal_default_instance();
  }
  static inline const GraphDebugInfo* internal_default_instance() {
    return reinterpret_cast<const GraphDebugInfo*>(
               &_GraphDebugInfo_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    3;

  friend void swap(GraphDebugInfo& a, GraphDebugInfo& b) {
    a.Swap(&b);
  }
  inline void Swap(GraphDebugInfo* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(GraphDebugInfo* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GraphDebugInfo* New() const final {
    return new GraphDebugInfo();
  }

  GraphDebugInfo* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GraphDebugInfo>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const GraphDebugInfo& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const GraphDebugInfo& from);
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
  void InternalSwap(GraphDebugInfo* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.GraphDebugInfo";
  }
  protected:
  explicit GraphDebugInfo(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  inline void RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef GraphDebugInfo_FileLineCol FileLineCol;
  typedef GraphDebugInfo_StackTrace StackTrace;

  // accessors -------------------------------------------------------

  enum : int {
    kFilesFieldNumber = 1,
    kTracesFieldNumber = 2,
  };
  // repeated string files = 1;
  int files_size() const;
  private:
  int _internal_files_size() const;
  public:
  void clear_files();
  const std::string& files(int index) const;
  std::string* mutable_files(int index);
  void set_files(int index, const std::string& value);
  void set_files(int index, std::string&& value);
  void set_files(int index, const char* value);
  void set_files(int index, const char* value, size_t size);
  std::string* add_files();
  void add_files(const std::string& value);
  void add_files(std::string&& value);
  void add_files(const char* value);
  void add_files(const char* value, size_t size);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>& files() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>* mutable_files();
  private:
  const std::string& _internal_files(int index) const;
  std::string* _internal_add_files();
  public:

  // map<string, .tensorflow.GraphDebugInfo.StackTrace> traces = 2;
  int traces_size() const;
  private:
  int _internal_traces_size() const;
  public:
  void clear_traces();
  private:
  const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >&
      _internal_traces() const;
  ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >*
      _internal_mutable_traces();
  public:
  const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >&
      traces() const;
  ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >*
      mutable_traces();

  // @@protoc_insertion_point(class_scope:tensorflow.GraphDebugInfo)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string> files_;
  ::PROTOBUF_NAMESPACE_ID::internal::MapField<
      GraphDebugInfo_TracesEntry_DoNotUse,
      std::string, ::tensorflow::GraphDebugInfo_StackTrace,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_MESSAGE> traces_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// GraphDebugInfo_FileLineCol

// int32 file_index = 1;
inline void GraphDebugInfo_FileLineCol::clear_file_index() {
  file_index_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 GraphDebugInfo_FileLineCol::_internal_file_index() const {
  return file_index_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 GraphDebugInfo_FileLineCol::file_index() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.FileLineCol.file_index)
  return _internal_file_index();
}
inline void GraphDebugInfo_FileLineCol::_internal_set_file_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  file_index_ = value;
}
inline void GraphDebugInfo_FileLineCol::set_file_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_file_index(value);
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.FileLineCol.file_index)
}

// int32 line = 2;
inline void GraphDebugInfo_FileLineCol::clear_line() {
  line_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 GraphDebugInfo_FileLineCol::_internal_line() const {
  return line_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 GraphDebugInfo_FileLineCol::line() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.FileLineCol.line)
  return _internal_line();
}
inline void GraphDebugInfo_FileLineCol::_internal_set_line(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  line_ = value;
}
inline void GraphDebugInfo_FileLineCol::set_line(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_line(value);
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.FileLineCol.line)
}

// int32 col = 3;
inline void GraphDebugInfo_FileLineCol::clear_col() {
  col_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 GraphDebugInfo_FileLineCol::_internal_col() const {
  return col_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 GraphDebugInfo_FileLineCol::col() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.FileLineCol.col)
  return _internal_col();
}
inline void GraphDebugInfo_FileLineCol::_internal_set_col(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  col_ = value;
}
inline void GraphDebugInfo_FileLineCol::set_col(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_col(value);
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.FileLineCol.col)
}

// string func = 4;
inline void GraphDebugInfo_FileLineCol::clear_func() {
  func_.ClearToEmpty();
}
inline const std::string& GraphDebugInfo_FileLineCol::func() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.FileLineCol.func)
  return _internal_func();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void GraphDebugInfo_FileLineCol::set_func(ArgT0&& arg0, ArgT... args) {
 
 func_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.FileLineCol.func)
}
inline std::string* GraphDebugInfo_FileLineCol::mutable_func() {
  std::string* _s = _internal_mutable_func();
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDebugInfo.FileLineCol.func)
  return _s;
}
inline const std::string& GraphDebugInfo_FileLineCol::_internal_func() const {
  return func_.Get();
}
inline void GraphDebugInfo_FileLineCol::_internal_set_func(const std::string& value) {
  
  func_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* GraphDebugInfo_FileLineCol::_internal_mutable_func() {
  
  return func_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* GraphDebugInfo_FileLineCol::release_func() {
  // @@protoc_insertion_point(field_release:tensorflow.GraphDebugInfo.FileLineCol.func)
  return func_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void GraphDebugInfo_FileLineCol::set_allocated_func(std::string* func) {
  if (func != nullptr) {
    
  } else {
    
  }
  func_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), func,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.GraphDebugInfo.FileLineCol.func)
}

// string code = 5;
inline void GraphDebugInfo_FileLineCol::clear_code() {
  code_.ClearToEmpty();
}
inline const std::string& GraphDebugInfo_FileLineCol::code() const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.FileLineCol.code)
  return _internal_code();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void GraphDebugInfo_FileLineCol::set_code(ArgT0&& arg0, ArgT... args) {
 
 code_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.FileLineCol.code)
}
inline std::string* GraphDebugInfo_FileLineCol::mutable_code() {
  std::string* _s = _internal_mutable_code();
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDebugInfo.FileLineCol.code)
  return _s;
}
inline const std::string& GraphDebugInfo_FileLineCol::_internal_code() const {
  return code_.Get();
}
inline void GraphDebugInfo_FileLineCol::_internal_set_code(const std::string& value) {
  
  code_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* GraphDebugInfo_FileLineCol::_internal_mutable_code() {
  
  return code_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* GraphDebugInfo_FileLineCol::release_code() {
  // @@protoc_insertion_point(field_release:tensorflow.GraphDebugInfo.FileLineCol.code)
  return code_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void GraphDebugInfo_FileLineCol::set_allocated_code(std::string* code) {
  if (code != nullptr) {
    
  } else {
    
  }
  code_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), code,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.GraphDebugInfo.FileLineCol.code)
}

// -------------------------------------------------------------------

// GraphDebugInfo_StackTrace

// repeated .tensorflow.GraphDebugInfo.FileLineCol file_line_cols = 1;
inline int GraphDebugInfo_StackTrace::_internal_file_line_cols_size() const {
  return file_line_cols_.size();
}
inline int GraphDebugInfo_StackTrace::file_line_cols_size() const {
  return _internal_file_line_cols_size();
}
inline void GraphDebugInfo_StackTrace::clear_file_line_cols() {
  file_line_cols_.Clear();
}
inline ::tensorflow::GraphDebugInfo_FileLineCol* GraphDebugInfo_StackTrace::mutable_file_line_cols(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDebugInfo.StackTrace.file_line_cols)
  return file_line_cols_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::GraphDebugInfo_FileLineCol >*
GraphDebugInfo_StackTrace::mutable_file_line_cols() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.GraphDebugInfo.StackTrace.file_line_cols)
  return &file_line_cols_;
}
inline const ::tensorflow::GraphDebugInfo_FileLineCol& GraphDebugInfo_StackTrace::_internal_file_line_cols(int index) const {
  return file_line_cols_.Get(index);
}
inline const ::tensorflow::GraphDebugInfo_FileLineCol& GraphDebugInfo_StackTrace::file_line_cols(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.StackTrace.file_line_cols)
  return _internal_file_line_cols(index);
}
inline ::tensorflow::GraphDebugInfo_FileLineCol* GraphDebugInfo_StackTrace::_internal_add_file_line_cols() {
  return file_line_cols_.Add();
}
inline ::tensorflow::GraphDebugInfo_FileLineCol* GraphDebugInfo_StackTrace::add_file_line_cols() {
  ::tensorflow::GraphDebugInfo_FileLineCol* _add = _internal_add_file_line_cols();
  // @@protoc_insertion_point(field_add:tensorflow.GraphDebugInfo.StackTrace.file_line_cols)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::GraphDebugInfo_FileLineCol >&
GraphDebugInfo_StackTrace::file_line_cols() const {
  // @@protoc_insertion_point(field_list:tensorflow.GraphDebugInfo.StackTrace.file_line_cols)
  return file_line_cols_;
}

// -------------------------------------------------------------------

// -------------------------------------------------------------------

// GraphDebugInfo

// repeated string files = 1;
inline int GraphDebugInfo::_internal_files_size() const {
  return files_.size();
}
inline int GraphDebugInfo::files_size() const {
  return _internal_files_size();
}
inline void GraphDebugInfo::clear_files() {
  files_.Clear();
}
inline std::string* GraphDebugInfo::add_files() {
  std::string* _s = _internal_add_files();
  // @@protoc_insertion_point(field_add_mutable:tensorflow.GraphDebugInfo.files)
  return _s;
}
inline const std::string& GraphDebugInfo::_internal_files(int index) const {
  return files_.Get(index);
}
inline const std::string& GraphDebugInfo::files(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.GraphDebugInfo.files)
  return _internal_files(index);
}
inline std::string* GraphDebugInfo::mutable_files(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.GraphDebugInfo.files)
  return files_.Mutable(index);
}
inline void GraphDebugInfo::set_files(int index, const std::string& value) {
  files_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.files)
}
inline void GraphDebugInfo::set_files(int index, std::string&& value) {
  files_.Mutable(index)->assign(std::move(value));
  // @@protoc_insertion_point(field_set:tensorflow.GraphDebugInfo.files)
}
inline void GraphDebugInfo::set_files(int index, const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  files_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:tensorflow.GraphDebugInfo.files)
}
inline void GraphDebugInfo::set_files(int index, const char* value, size_t size) {
  files_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:tensorflow.GraphDebugInfo.files)
}
inline std::string* GraphDebugInfo::_internal_add_files() {
  return files_.Add();
}
inline void GraphDebugInfo::add_files(const std::string& value) {
  files_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:tensorflow.GraphDebugInfo.files)
}
inline void GraphDebugInfo::add_files(std::string&& value) {
  files_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:tensorflow.GraphDebugInfo.files)
}
inline void GraphDebugInfo::add_files(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  files_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:tensorflow.GraphDebugInfo.files)
}
inline void GraphDebugInfo::add_files(const char* value, size_t size) {
  files_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:tensorflow.GraphDebugInfo.files)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>&
GraphDebugInfo::files() const {
  // @@protoc_insertion_point(field_list:tensorflow.GraphDebugInfo.files)
  return files_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>*
GraphDebugInfo::mutable_files() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.GraphDebugInfo.files)
  return &files_;
}

// map<string, .tensorflow.GraphDebugInfo.StackTrace> traces = 2;
inline int GraphDebugInfo::_internal_traces_size() const {
  return traces_.size();
}
inline int GraphDebugInfo::traces_size() const {
  return _internal_traces_size();
}
inline void GraphDebugInfo::clear_traces() {
  traces_.Clear();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >&
GraphDebugInfo::_internal_traces() const {
  return traces_.GetMap();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >&
GraphDebugInfo::traces() const {
  // @@protoc_insertion_point(field_map:tensorflow.GraphDebugInfo.traces)
  return _internal_traces();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >*
GraphDebugInfo::_internal_mutable_traces() {
  return traces_.MutableMap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< std::string, ::tensorflow::GraphDebugInfo_StackTrace >*
GraphDebugInfo::mutable_traces() {
  // @@protoc_insertion_point(field_mutable_map:tensorflow.GraphDebugInfo.traces)
  return _internal_mutable_traces();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fgraph_5fdebug_5finfo_2eproto
