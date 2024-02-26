// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/tensorflow_server.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto

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
#include "tensorflow/core/protobuf/cluster.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/device_filters.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto;
namespace tensorflow {
class ServerDef;
struct ServerDefDefaultTypeInternal;
extern ServerDefDefaultTypeInternal _ServerDef_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::ServerDef* Arena::CreateMaybeMessage<::tensorflow::ServerDef>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class ServerDef final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.ServerDef) */ {
 public:
  inline ServerDef() : ServerDef(nullptr) {}
  ~ServerDef() override;
  explicit constexpr ServerDef(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ServerDef(const ServerDef& from);
  ServerDef(ServerDef&& from) noexcept
    : ServerDef() {
    *this = ::std::move(from);
  }

  inline ServerDef& operator=(const ServerDef& from) {
    CopyFrom(from);
    return *this;
  }
  inline ServerDef& operator=(ServerDef&& from) noexcept {
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
  static const ServerDef& default_instance() {
    return *internal_default_instance();
  }
  static inline const ServerDef* internal_default_instance() {
    return reinterpret_cast<const ServerDef*>(
               &_ServerDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ServerDef& a, ServerDef& b) {
    a.Swap(&b);
  }
  inline void Swap(ServerDef* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(ServerDef* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ServerDef* New() const final {
    return new ServerDef();
  }

  ServerDef* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ServerDef>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const ServerDef& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const ServerDef& from);
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
  void InternalSwap(ServerDef* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.ServerDef";
  }
  protected:
  explicit ServerDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kJobNameFieldNumber = 2,
    kProtocolFieldNumber = 5,
    kClusterFieldNumber = 1,
    kDefaultSessionConfigFieldNumber = 4,
    kClusterDeviceFiltersFieldNumber = 7,
    kTaskIndexFieldNumber = 3,
    kPortFieldNumber = 6,
  };
  // string job_name = 2;
  void clear_job_name();
  const std::string& job_name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_job_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_job_name();
  PROTOBUF_MUST_USE_RESULT std::string* release_job_name();
  void set_allocated_job_name(std::string* job_name);
  private:
  const std::string& _internal_job_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_job_name(const std::string& value);
  std::string* _internal_mutable_job_name();
  public:

  // string protocol = 5;
  void clear_protocol();
  const std::string& protocol() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_protocol(ArgT0&& arg0, ArgT... args);
  std::string* mutable_protocol();
  PROTOBUF_MUST_USE_RESULT std::string* release_protocol();
  void set_allocated_protocol(std::string* protocol);
  private:
  const std::string& _internal_protocol() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_protocol(const std::string& value);
  std::string* _internal_mutable_protocol();
  public:

  // .tensorflow.ClusterDef cluster = 1;
  bool has_cluster() const;
  private:
  bool _internal_has_cluster() const;
  public:
  void clear_cluster();
  const ::tensorflow::ClusterDef& cluster() const;
  PROTOBUF_MUST_USE_RESULT ::tensorflow::ClusterDef* release_cluster();
  ::tensorflow::ClusterDef* mutable_cluster();
  void set_allocated_cluster(::tensorflow::ClusterDef* cluster);
  private:
  const ::tensorflow::ClusterDef& _internal_cluster() const;
  ::tensorflow::ClusterDef* _internal_mutable_cluster();
  public:
  void unsafe_arena_set_allocated_cluster(
      ::tensorflow::ClusterDef* cluster);
  ::tensorflow::ClusterDef* unsafe_arena_release_cluster();

  // .tensorflow.ConfigProto default_session_config = 4;
  bool has_default_session_config() const;
  private:
  bool _internal_has_default_session_config() const;
  public:
  void clear_default_session_config();
  const ::tensorflow::ConfigProto& default_session_config() const;
  PROTOBUF_MUST_USE_RESULT ::tensorflow::ConfigProto* release_default_session_config();
  ::tensorflow::ConfigProto* mutable_default_session_config();
  void set_allocated_default_session_config(::tensorflow::ConfigProto* default_session_config);
  private:
  const ::tensorflow::ConfigProto& _internal_default_session_config() const;
  ::tensorflow::ConfigProto* _internal_mutable_default_session_config();
  public:
  void unsafe_arena_set_allocated_default_session_config(
      ::tensorflow::ConfigProto* default_session_config);
  ::tensorflow::ConfigProto* unsafe_arena_release_default_session_config();

  // .tensorflow.ClusterDeviceFilters cluster_device_filters = 7;
  bool has_cluster_device_filters() const;
  private:
  bool _internal_has_cluster_device_filters() const;
  public:
  void clear_cluster_device_filters();
  const ::tensorflow::ClusterDeviceFilters& cluster_device_filters() const;
  PROTOBUF_MUST_USE_RESULT ::tensorflow::ClusterDeviceFilters* release_cluster_device_filters();
  ::tensorflow::ClusterDeviceFilters* mutable_cluster_device_filters();
  void set_allocated_cluster_device_filters(::tensorflow::ClusterDeviceFilters* cluster_device_filters);
  private:
  const ::tensorflow::ClusterDeviceFilters& _internal_cluster_device_filters() const;
  ::tensorflow::ClusterDeviceFilters* _internal_mutable_cluster_device_filters();
  public:
  void unsafe_arena_set_allocated_cluster_device_filters(
      ::tensorflow::ClusterDeviceFilters* cluster_device_filters);
  ::tensorflow::ClusterDeviceFilters* unsafe_arena_release_cluster_device_filters();

  // int32 task_index = 3;
  void clear_task_index();
  ::PROTOBUF_NAMESPACE_ID::int32 task_index() const;
  void set_task_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_task_index() const;
  void _internal_set_task_index(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // int32 port = 6;
  void clear_port();
  ::PROTOBUF_NAMESPACE_ID::int32 port() const;
  void set_port(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_port() const;
  void _internal_set_port(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.ServerDef)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr job_name_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr protocol_;
  ::tensorflow::ClusterDef* cluster_;
  ::tensorflow::ConfigProto* default_session_config_;
  ::tensorflow::ClusterDeviceFilters* cluster_device_filters_;
  ::PROTOBUF_NAMESPACE_ID::int32 task_index_;
  ::PROTOBUF_NAMESPACE_ID::int32 port_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ServerDef

// .tensorflow.ClusterDef cluster = 1;
inline bool ServerDef::_internal_has_cluster() const {
  return this != internal_default_instance() && cluster_ != nullptr;
}
inline bool ServerDef::has_cluster() const {
  return _internal_has_cluster();
}
inline const ::tensorflow::ClusterDef& ServerDef::_internal_cluster() const {
  const ::tensorflow::ClusterDef* p = cluster_;
  return p != nullptr ? *p : reinterpret_cast<const ::tensorflow::ClusterDef&>(
      ::tensorflow::_ClusterDef_default_instance_);
}
inline const ::tensorflow::ClusterDef& ServerDef::cluster() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.cluster)
  return _internal_cluster();
}
inline void ServerDef::unsafe_arena_set_allocated_cluster(
    ::tensorflow::ClusterDef* cluster) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(cluster_);
  }
  cluster_ = cluster;
  if (cluster) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.ServerDef.cluster)
}
inline ::tensorflow::ClusterDef* ServerDef::release_cluster() {
  
  ::tensorflow::ClusterDef* temp = cluster_;
  cluster_ = nullptr;
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
inline ::tensorflow::ClusterDef* ServerDef::unsafe_arena_release_cluster() {
  // @@protoc_insertion_point(field_release:tensorflow.ServerDef.cluster)
  
  ::tensorflow::ClusterDef* temp = cluster_;
  cluster_ = nullptr;
  return temp;
}
inline ::tensorflow::ClusterDef* ServerDef::_internal_mutable_cluster() {
  
  if (cluster_ == nullptr) {
    auto* p = CreateMaybeMessage<::tensorflow::ClusterDef>(GetArenaForAllocation());
    cluster_ = p;
  }
  return cluster_;
}
inline ::tensorflow::ClusterDef* ServerDef::mutable_cluster() {
  ::tensorflow::ClusterDef* _msg = _internal_mutable_cluster();
  // @@protoc_insertion_point(field_mutable:tensorflow.ServerDef.cluster)
  return _msg;
}
inline void ServerDef::set_allocated_cluster(::tensorflow::ClusterDef* cluster) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(cluster_);
  }
  if (cluster) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<
            ::PROTOBUF_NAMESPACE_ID::MessageLite>::GetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(cluster));
    if (message_arena != submessage_arena) {
      cluster = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, cluster, submessage_arena);
    }
    
  } else {
    
  }
  cluster_ = cluster;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ServerDef.cluster)
}

// string job_name = 2;
inline void ServerDef::clear_job_name() {
  job_name_.ClearToEmpty();
}
inline const std::string& ServerDef::job_name() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.job_name)
  return _internal_job_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ServerDef::set_job_name(ArgT0&& arg0, ArgT... args) {
 
 job_name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.ServerDef.job_name)
}
inline std::string* ServerDef::mutable_job_name() {
  std::string* _s = _internal_mutable_job_name();
  // @@protoc_insertion_point(field_mutable:tensorflow.ServerDef.job_name)
  return _s;
}
inline const std::string& ServerDef::_internal_job_name() const {
  return job_name_.Get();
}
inline void ServerDef::_internal_set_job_name(const std::string& value) {
  
  job_name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ServerDef::_internal_mutable_job_name() {
  
  return job_name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ServerDef::release_job_name() {
  // @@protoc_insertion_point(field_release:tensorflow.ServerDef.job_name)
  return job_name_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ServerDef::set_allocated_job_name(std::string* job_name) {
  if (job_name != nullptr) {
    
  } else {
    
  }
  job_name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), job_name,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ServerDef.job_name)
}

// int32 task_index = 3;
inline void ServerDef::clear_task_index() {
  task_index_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ServerDef::_internal_task_index() const {
  return task_index_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ServerDef::task_index() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.task_index)
  return _internal_task_index();
}
inline void ServerDef::_internal_set_task_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  task_index_ = value;
}
inline void ServerDef::set_task_index(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_task_index(value);
  // @@protoc_insertion_point(field_set:tensorflow.ServerDef.task_index)
}

// .tensorflow.ConfigProto default_session_config = 4;
inline bool ServerDef::_internal_has_default_session_config() const {
  return this != internal_default_instance() && default_session_config_ != nullptr;
}
inline bool ServerDef::has_default_session_config() const {
  return _internal_has_default_session_config();
}
inline const ::tensorflow::ConfigProto& ServerDef::_internal_default_session_config() const {
  const ::tensorflow::ConfigProto* p = default_session_config_;
  return p != nullptr ? *p : reinterpret_cast<const ::tensorflow::ConfigProto&>(
      ::tensorflow::_ConfigProto_default_instance_);
}
inline const ::tensorflow::ConfigProto& ServerDef::default_session_config() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.default_session_config)
  return _internal_default_session_config();
}
inline void ServerDef::unsafe_arena_set_allocated_default_session_config(
    ::tensorflow::ConfigProto* default_session_config) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(default_session_config_);
  }
  default_session_config_ = default_session_config;
  if (default_session_config) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.ServerDef.default_session_config)
}
inline ::tensorflow::ConfigProto* ServerDef::release_default_session_config() {
  
  ::tensorflow::ConfigProto* temp = default_session_config_;
  default_session_config_ = nullptr;
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
inline ::tensorflow::ConfigProto* ServerDef::unsafe_arena_release_default_session_config() {
  // @@protoc_insertion_point(field_release:tensorflow.ServerDef.default_session_config)
  
  ::tensorflow::ConfigProto* temp = default_session_config_;
  default_session_config_ = nullptr;
  return temp;
}
inline ::tensorflow::ConfigProto* ServerDef::_internal_mutable_default_session_config() {
  
  if (default_session_config_ == nullptr) {
    auto* p = CreateMaybeMessage<::tensorflow::ConfigProto>(GetArenaForAllocation());
    default_session_config_ = p;
  }
  return default_session_config_;
}
inline ::tensorflow::ConfigProto* ServerDef::mutable_default_session_config() {
  ::tensorflow::ConfigProto* _msg = _internal_mutable_default_session_config();
  // @@protoc_insertion_point(field_mutable:tensorflow.ServerDef.default_session_config)
  return _msg;
}
inline void ServerDef::set_allocated_default_session_config(::tensorflow::ConfigProto* default_session_config) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(default_session_config_);
  }
  if (default_session_config) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<
            ::PROTOBUF_NAMESPACE_ID::MessageLite>::GetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(default_session_config));
    if (message_arena != submessage_arena) {
      default_session_config = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, default_session_config, submessage_arena);
    }
    
  } else {
    
  }
  default_session_config_ = default_session_config;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ServerDef.default_session_config)
}

// string protocol = 5;
inline void ServerDef::clear_protocol() {
  protocol_.ClearToEmpty();
}
inline const std::string& ServerDef::protocol() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.protocol)
  return _internal_protocol();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void ServerDef::set_protocol(ArgT0&& arg0, ArgT... args) {
 
 protocol_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.ServerDef.protocol)
}
inline std::string* ServerDef::mutable_protocol() {
  std::string* _s = _internal_mutable_protocol();
  // @@protoc_insertion_point(field_mutable:tensorflow.ServerDef.protocol)
  return _s;
}
inline const std::string& ServerDef::_internal_protocol() const {
  return protocol_.Get();
}
inline void ServerDef::_internal_set_protocol(const std::string& value) {
  
  protocol_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* ServerDef::_internal_mutable_protocol() {
  
  return protocol_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* ServerDef::release_protocol() {
  // @@protoc_insertion_point(field_release:tensorflow.ServerDef.protocol)
  return protocol_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void ServerDef::set_allocated_protocol(std::string* protocol) {
  if (protocol != nullptr) {
    
  } else {
    
  }
  protocol_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), protocol,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ServerDef.protocol)
}

// int32 port = 6;
inline void ServerDef::clear_port() {
  port_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ServerDef::_internal_port() const {
  return port_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 ServerDef::port() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.port)
  return _internal_port();
}
inline void ServerDef::_internal_set_port(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  port_ = value;
}
inline void ServerDef::set_port(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_port(value);
  // @@protoc_insertion_point(field_set:tensorflow.ServerDef.port)
}

// .tensorflow.ClusterDeviceFilters cluster_device_filters = 7;
inline bool ServerDef::_internal_has_cluster_device_filters() const {
  return this != internal_default_instance() && cluster_device_filters_ != nullptr;
}
inline bool ServerDef::has_cluster_device_filters() const {
  return _internal_has_cluster_device_filters();
}
inline const ::tensorflow::ClusterDeviceFilters& ServerDef::_internal_cluster_device_filters() const {
  const ::tensorflow::ClusterDeviceFilters* p = cluster_device_filters_;
  return p != nullptr ? *p : reinterpret_cast<const ::tensorflow::ClusterDeviceFilters&>(
      ::tensorflow::_ClusterDeviceFilters_default_instance_);
}
inline const ::tensorflow::ClusterDeviceFilters& ServerDef::cluster_device_filters() const {
  // @@protoc_insertion_point(field_get:tensorflow.ServerDef.cluster_device_filters)
  return _internal_cluster_device_filters();
}
inline void ServerDef::unsafe_arena_set_allocated_cluster_device_filters(
    ::tensorflow::ClusterDeviceFilters* cluster_device_filters) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(cluster_device_filters_);
  }
  cluster_device_filters_ = cluster_device_filters;
  if (cluster_device_filters) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.ServerDef.cluster_device_filters)
}
inline ::tensorflow::ClusterDeviceFilters* ServerDef::release_cluster_device_filters() {
  
  ::tensorflow::ClusterDeviceFilters* temp = cluster_device_filters_;
  cluster_device_filters_ = nullptr;
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
inline ::tensorflow::ClusterDeviceFilters* ServerDef::unsafe_arena_release_cluster_device_filters() {
  // @@protoc_insertion_point(field_release:tensorflow.ServerDef.cluster_device_filters)
  
  ::tensorflow::ClusterDeviceFilters* temp = cluster_device_filters_;
  cluster_device_filters_ = nullptr;
  return temp;
}
inline ::tensorflow::ClusterDeviceFilters* ServerDef::_internal_mutable_cluster_device_filters() {
  
  if (cluster_device_filters_ == nullptr) {
    auto* p = CreateMaybeMessage<::tensorflow::ClusterDeviceFilters>(GetArenaForAllocation());
    cluster_device_filters_ = p;
  }
  return cluster_device_filters_;
}
inline ::tensorflow::ClusterDeviceFilters* ServerDef::mutable_cluster_device_filters() {
  ::tensorflow::ClusterDeviceFilters* _msg = _internal_mutable_cluster_device_filters();
  // @@protoc_insertion_point(field_mutable:tensorflow.ServerDef.cluster_device_filters)
  return _msg;
}
inline void ServerDef::set_allocated_cluster_device_filters(::tensorflow::ClusterDeviceFilters* cluster_device_filters) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(cluster_device_filters_);
  }
  if (cluster_device_filters) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper<
            ::PROTOBUF_NAMESPACE_ID::MessageLite>::GetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(cluster_device_filters));
    if (message_arena != submessage_arena) {
      cluster_device_filters = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, cluster_device_filters, submessage_arena);
    }
    
  } else {
    
  }
  cluster_device_filters_ = cluster_device_filters;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.ServerDef.cluster_device_filters)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ftensorflow_5fserver_2eproto
