// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/coordination_config.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto {
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
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto;
namespace tensorflow {
class CoordinatedJob;
struct CoordinatedJobDefaultTypeInternal;
extern CoordinatedJobDefaultTypeInternal _CoordinatedJob_default_instance_;
class CoordinationServiceConfig;
struct CoordinationServiceConfigDefaultTypeInternal;
extern CoordinationServiceConfigDefaultTypeInternal _CoordinationServiceConfig_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::CoordinatedJob* Arena::CreateMaybeMessage<::tensorflow::CoordinatedJob>(Arena*);
template<> ::tensorflow::CoordinationServiceConfig* Arena::CreateMaybeMessage<::tensorflow::CoordinationServiceConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class CoordinatedJob final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.CoordinatedJob) */ {
 public:
  inline CoordinatedJob() : CoordinatedJob(nullptr) {}
  ~CoordinatedJob() override;
  explicit constexpr CoordinatedJob(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  CoordinatedJob(const CoordinatedJob& from);
  CoordinatedJob(CoordinatedJob&& from) noexcept
    : CoordinatedJob() {
    *this = ::std::move(from);
  }

  inline CoordinatedJob& operator=(const CoordinatedJob& from) {
    CopyFrom(from);
    return *this;
  }
  inline CoordinatedJob& operator=(CoordinatedJob&& from) noexcept {
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
  static const CoordinatedJob& default_instance() {
    return *internal_default_instance();
  }
  static inline const CoordinatedJob* internal_default_instance() {
    return reinterpret_cast<const CoordinatedJob*>(
               &_CoordinatedJob_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(CoordinatedJob& a, CoordinatedJob& b) {
    a.Swap(&b);
  }
  inline void Swap(CoordinatedJob* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(CoordinatedJob* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline CoordinatedJob* New() const final {
    return new CoordinatedJob();
  }

  CoordinatedJob* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<CoordinatedJob>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const CoordinatedJob& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const CoordinatedJob& from);
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
  void InternalSwap(CoordinatedJob* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.CoordinatedJob";
  }
  protected:
  explicit CoordinatedJob(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kNameFieldNumber = 1,
    kNumTasksFieldNumber = 2,
  };
  // string name = 1;
  void clear_name();
  const std::string& name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_name();
  PROTOBUF_MUST_USE_RESULT std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // int32 num_tasks = 2;
  void clear_num_tasks();
  ::PROTOBUF_NAMESPACE_ID::int32 num_tasks() const;
  void set_num_tasks(::PROTOBUF_NAMESPACE_ID::int32 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int32 _internal_num_tasks() const;
  void _internal_set_num_tasks(::PROTOBUF_NAMESPACE_ID::int32 value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.CoordinatedJob)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  ::PROTOBUF_NAMESPACE_ID::int32 num_tasks_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto;
};
// -------------------------------------------------------------------

class CoordinationServiceConfig final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.CoordinationServiceConfig) */ {
 public:
  inline CoordinationServiceConfig() : CoordinationServiceConfig(nullptr) {}
  ~CoordinationServiceConfig() override;
  explicit constexpr CoordinationServiceConfig(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  CoordinationServiceConfig(const CoordinationServiceConfig& from);
  CoordinationServiceConfig(CoordinationServiceConfig&& from) noexcept
    : CoordinationServiceConfig() {
    *this = ::std::move(from);
  }

  inline CoordinationServiceConfig& operator=(const CoordinationServiceConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline CoordinationServiceConfig& operator=(CoordinationServiceConfig&& from) noexcept {
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
  static const CoordinationServiceConfig& default_instance() {
    return *internal_default_instance();
  }
  static inline const CoordinationServiceConfig* internal_default_instance() {
    return reinterpret_cast<const CoordinationServiceConfig*>(
               &_CoordinationServiceConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(CoordinationServiceConfig& a, CoordinationServiceConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(CoordinationServiceConfig* other) {
    if (other == this) return;
    if (GetOwningArena() == other->GetOwningArena()) {
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(CoordinationServiceConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline CoordinationServiceConfig* New() const final {
    return new CoordinationServiceConfig();
  }

  CoordinationServiceConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<CoordinationServiceConfig>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const CoordinationServiceConfig& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom(const CoordinationServiceConfig& from);
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
  void InternalSwap(CoordinationServiceConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.CoordinationServiceConfig";
  }
  protected:
  explicit CoordinationServiceConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
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
    kRecoverableJobsFieldNumber = 9,
    kCoordinatedJobListFieldNumber = 10,
    kServiceTypeFieldNumber = 1,
    kServiceLeaderFieldNumber = 2,
    kClusterRegisterTimeoutInMsFieldNumber = 4,
    kHeartbeatTimeoutInMsFieldNumber = 5,
    kShutdownBarrierTimeoutInMsFieldNumber = 7,
    kEnableHealthCheckFieldNumber = 3,
    kAgentDestructionWithoutShutdownFieldNumber = 8,
  };
  // repeated string recoverable_jobs = 9;
  int recoverable_jobs_size() const;
  private:
  int _internal_recoverable_jobs_size() const;
  public:
  void clear_recoverable_jobs();
  const std::string& recoverable_jobs(int index) const;
  std::string* mutable_recoverable_jobs(int index);
  void set_recoverable_jobs(int index, const std::string& value);
  void set_recoverable_jobs(int index, std::string&& value);
  void set_recoverable_jobs(int index, const char* value);
  void set_recoverable_jobs(int index, const char* value, size_t size);
  std::string* add_recoverable_jobs();
  void add_recoverable_jobs(const std::string& value);
  void add_recoverable_jobs(std::string&& value);
  void add_recoverable_jobs(const char* value);
  void add_recoverable_jobs(const char* value, size_t size);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>& recoverable_jobs() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>* mutable_recoverable_jobs();
  private:
  const std::string& _internal_recoverable_jobs(int index) const;
  std::string* _internal_add_recoverable_jobs();
  public:

  // repeated .tensorflow.CoordinatedJob coordinated_job_list = 10;
  int coordinated_job_list_size() const;
  private:
  int _internal_coordinated_job_list_size() const;
  public:
  void clear_coordinated_job_list();
  ::tensorflow::CoordinatedJob* mutable_coordinated_job_list(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::CoordinatedJob >*
      mutable_coordinated_job_list();
  private:
  const ::tensorflow::CoordinatedJob& _internal_coordinated_job_list(int index) const;
  ::tensorflow::CoordinatedJob* _internal_add_coordinated_job_list();
  public:
  const ::tensorflow::CoordinatedJob& coordinated_job_list(int index) const;
  ::tensorflow::CoordinatedJob* add_coordinated_job_list();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::CoordinatedJob >&
      coordinated_job_list() const;

  // string service_type = 1;
  void clear_service_type();
  const std::string& service_type() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_service_type(ArgT0&& arg0, ArgT... args);
  std::string* mutable_service_type();
  PROTOBUF_MUST_USE_RESULT std::string* release_service_type();
  void set_allocated_service_type(std::string* service_type);
  private:
  const std::string& _internal_service_type() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_service_type(const std::string& value);
  std::string* _internal_mutable_service_type();
  public:

  // string service_leader = 2;
  void clear_service_leader();
  const std::string& service_leader() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_service_leader(ArgT0&& arg0, ArgT... args);
  std::string* mutable_service_leader();
  PROTOBUF_MUST_USE_RESULT std::string* release_service_leader();
  void set_allocated_service_leader(std::string* service_leader);
  private:
  const std::string& _internal_service_leader() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_service_leader(const std::string& value);
  std::string* _internal_mutable_service_leader();
  public:

  // int64 cluster_register_timeout_in_ms = 4;
  void clear_cluster_register_timeout_in_ms();
  ::PROTOBUF_NAMESPACE_ID::int64 cluster_register_timeout_in_ms() const;
  void set_cluster_register_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_cluster_register_timeout_in_ms() const;
  void _internal_set_cluster_register_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 heartbeat_timeout_in_ms = 5;
  void clear_heartbeat_timeout_in_ms();
  ::PROTOBUF_NAMESPACE_ID::int64 heartbeat_timeout_in_ms() const;
  void set_heartbeat_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_heartbeat_timeout_in_ms() const;
  void _internal_set_heartbeat_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // int64 shutdown_barrier_timeout_in_ms = 7;
  void clear_shutdown_barrier_timeout_in_ms();
  ::PROTOBUF_NAMESPACE_ID::int64 shutdown_barrier_timeout_in_ms() const;
  void set_shutdown_barrier_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value);
  private:
  ::PROTOBUF_NAMESPACE_ID::int64 _internal_shutdown_barrier_timeout_in_ms() const;
  void _internal_set_shutdown_barrier_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value);
  public:

  // bool enable_health_check = 3;
  void clear_enable_health_check();
  bool enable_health_check() const;
  void set_enable_health_check(bool value);
  private:
  bool _internal_enable_health_check() const;
  void _internal_set_enable_health_check(bool value);
  public:

  // bool agent_destruction_without_shutdown = 8;
  void clear_agent_destruction_without_shutdown();
  bool agent_destruction_without_shutdown() const;
  void set_agent_destruction_without_shutdown(bool value);
  private:
  bool _internal_agent_destruction_without_shutdown() const;
  void _internal_set_agent_destruction_without_shutdown(bool value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.CoordinationServiceConfig)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string> recoverable_jobs_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::CoordinatedJob > coordinated_job_list_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr service_type_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr service_leader_;
  ::PROTOBUF_NAMESPACE_ID::int64 cluster_register_timeout_in_ms_;
  ::PROTOBUF_NAMESPACE_ID::int64 heartbeat_timeout_in_ms_;
  ::PROTOBUF_NAMESPACE_ID::int64 shutdown_barrier_timeout_in_ms_;
  bool enable_health_check_;
  bool agent_destruction_without_shutdown_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CoordinatedJob

// string name = 1;
inline void CoordinatedJob::clear_name() {
  name_.ClearToEmpty();
}
inline const std::string& CoordinatedJob::name() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinatedJob.name)
  return _internal_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void CoordinatedJob::set_name(ArgT0&& arg0, ArgT... args) {
 
 name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.CoordinatedJob.name)
}
inline std::string* CoordinatedJob::mutable_name() {
  std::string* _s = _internal_mutable_name();
  // @@protoc_insertion_point(field_mutable:tensorflow.CoordinatedJob.name)
  return _s;
}
inline const std::string& CoordinatedJob::_internal_name() const {
  return name_.Get();
}
inline void CoordinatedJob::_internal_set_name(const std::string& value) {
  
  name_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* CoordinatedJob::_internal_mutable_name() {
  
  return name_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* CoordinatedJob::release_name() {
  // @@protoc_insertion_point(field_release:tensorflow.CoordinatedJob.name)
  return name_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void CoordinatedJob::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    
  } else {
    
  }
  name_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.CoordinatedJob.name)
}

// int32 num_tasks = 2;
inline void CoordinatedJob::clear_num_tasks() {
  num_tasks_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 CoordinatedJob::_internal_num_tasks() const {
  return num_tasks_;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 CoordinatedJob::num_tasks() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinatedJob.num_tasks)
  return _internal_num_tasks();
}
inline void CoordinatedJob::_internal_set_num_tasks(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  num_tasks_ = value;
}
inline void CoordinatedJob::set_num_tasks(::PROTOBUF_NAMESPACE_ID::int32 value) {
  _internal_set_num_tasks(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinatedJob.num_tasks)
}

// -------------------------------------------------------------------

// CoordinationServiceConfig

// string service_type = 1;
inline void CoordinationServiceConfig::clear_service_type() {
  service_type_.ClearToEmpty();
}
inline const std::string& CoordinationServiceConfig::service_type() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.service_type)
  return _internal_service_type();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void CoordinationServiceConfig::set_service_type(ArgT0&& arg0, ArgT... args) {
 
 service_type_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.service_type)
}
inline std::string* CoordinationServiceConfig::mutable_service_type() {
  std::string* _s = _internal_mutable_service_type();
  // @@protoc_insertion_point(field_mutable:tensorflow.CoordinationServiceConfig.service_type)
  return _s;
}
inline const std::string& CoordinationServiceConfig::_internal_service_type() const {
  return service_type_.Get();
}
inline void CoordinationServiceConfig::_internal_set_service_type(const std::string& value) {
  
  service_type_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* CoordinationServiceConfig::_internal_mutable_service_type() {
  
  return service_type_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* CoordinationServiceConfig::release_service_type() {
  // @@protoc_insertion_point(field_release:tensorflow.CoordinationServiceConfig.service_type)
  return service_type_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void CoordinationServiceConfig::set_allocated_service_type(std::string* service_type) {
  if (service_type != nullptr) {
    
  } else {
    
  }
  service_type_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), service_type,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.CoordinationServiceConfig.service_type)
}

// string service_leader = 2;
inline void CoordinationServiceConfig::clear_service_leader() {
  service_leader_.ClearToEmpty();
}
inline const std::string& CoordinationServiceConfig::service_leader() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.service_leader)
  return _internal_service_leader();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void CoordinationServiceConfig::set_service_leader(ArgT0&& arg0, ArgT... args) {
 
 service_leader_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.service_leader)
}
inline std::string* CoordinationServiceConfig::mutable_service_leader() {
  std::string* _s = _internal_mutable_service_leader();
  // @@protoc_insertion_point(field_mutable:tensorflow.CoordinationServiceConfig.service_leader)
  return _s;
}
inline const std::string& CoordinationServiceConfig::_internal_service_leader() const {
  return service_leader_.Get();
}
inline void CoordinationServiceConfig::_internal_set_service_leader(const std::string& value) {
  
  service_leader_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, value, GetArenaForAllocation());
}
inline std::string* CoordinationServiceConfig::_internal_mutable_service_leader() {
  
  return service_leader_.Mutable(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, GetArenaForAllocation());
}
inline std::string* CoordinationServiceConfig::release_service_leader() {
  // @@protoc_insertion_point(field_release:tensorflow.CoordinationServiceConfig.service_leader)
  return service_leader_.Release(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaForAllocation());
}
inline void CoordinationServiceConfig::set_allocated_service_leader(std::string* service_leader) {
  if (service_leader != nullptr) {
    
  } else {
    
  }
  service_leader_.SetAllocated(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), service_leader,
      GetArenaForAllocation());
  // @@protoc_insertion_point(field_set_allocated:tensorflow.CoordinationServiceConfig.service_leader)
}

// bool enable_health_check = 3;
inline void CoordinationServiceConfig::clear_enable_health_check() {
  enable_health_check_ = false;
}
inline bool CoordinationServiceConfig::_internal_enable_health_check() const {
  return enable_health_check_;
}
inline bool CoordinationServiceConfig::enable_health_check() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.enable_health_check)
  return _internal_enable_health_check();
}
inline void CoordinationServiceConfig::_internal_set_enable_health_check(bool value) {
  
  enable_health_check_ = value;
}
inline void CoordinationServiceConfig::set_enable_health_check(bool value) {
  _internal_set_enable_health_check(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.enable_health_check)
}

// int64 cluster_register_timeout_in_ms = 4;
inline void CoordinationServiceConfig::clear_cluster_register_timeout_in_ms() {
  cluster_register_timeout_in_ms_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 CoordinationServiceConfig::_internal_cluster_register_timeout_in_ms() const {
  return cluster_register_timeout_in_ms_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 CoordinationServiceConfig::cluster_register_timeout_in_ms() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.cluster_register_timeout_in_ms)
  return _internal_cluster_register_timeout_in_ms();
}
inline void CoordinationServiceConfig::_internal_set_cluster_register_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  cluster_register_timeout_in_ms_ = value;
}
inline void CoordinationServiceConfig::set_cluster_register_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_cluster_register_timeout_in_ms(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.cluster_register_timeout_in_ms)
}

// int64 heartbeat_timeout_in_ms = 5;
inline void CoordinationServiceConfig::clear_heartbeat_timeout_in_ms() {
  heartbeat_timeout_in_ms_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 CoordinationServiceConfig::_internal_heartbeat_timeout_in_ms() const {
  return heartbeat_timeout_in_ms_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 CoordinationServiceConfig::heartbeat_timeout_in_ms() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.heartbeat_timeout_in_ms)
  return _internal_heartbeat_timeout_in_ms();
}
inline void CoordinationServiceConfig::_internal_set_heartbeat_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  heartbeat_timeout_in_ms_ = value;
}
inline void CoordinationServiceConfig::set_heartbeat_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_heartbeat_timeout_in_ms(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.heartbeat_timeout_in_ms)
}

// repeated .tensorflow.CoordinatedJob coordinated_job_list = 10;
inline int CoordinationServiceConfig::_internal_coordinated_job_list_size() const {
  return coordinated_job_list_.size();
}
inline int CoordinationServiceConfig::coordinated_job_list_size() const {
  return _internal_coordinated_job_list_size();
}
inline void CoordinationServiceConfig::clear_coordinated_job_list() {
  coordinated_job_list_.Clear();
}
inline ::tensorflow::CoordinatedJob* CoordinationServiceConfig::mutable_coordinated_job_list(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.CoordinationServiceConfig.coordinated_job_list)
  return coordinated_job_list_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::CoordinatedJob >*
CoordinationServiceConfig::mutable_coordinated_job_list() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.CoordinationServiceConfig.coordinated_job_list)
  return &coordinated_job_list_;
}
inline const ::tensorflow::CoordinatedJob& CoordinationServiceConfig::_internal_coordinated_job_list(int index) const {
  return coordinated_job_list_.Get(index);
}
inline const ::tensorflow::CoordinatedJob& CoordinationServiceConfig::coordinated_job_list(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.coordinated_job_list)
  return _internal_coordinated_job_list(index);
}
inline ::tensorflow::CoordinatedJob* CoordinationServiceConfig::_internal_add_coordinated_job_list() {
  return coordinated_job_list_.Add();
}
inline ::tensorflow::CoordinatedJob* CoordinationServiceConfig::add_coordinated_job_list() {
  ::tensorflow::CoordinatedJob* _add = _internal_add_coordinated_job_list();
  // @@protoc_insertion_point(field_add:tensorflow.CoordinationServiceConfig.coordinated_job_list)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::CoordinatedJob >&
CoordinationServiceConfig::coordinated_job_list() const {
  // @@protoc_insertion_point(field_list:tensorflow.CoordinationServiceConfig.coordinated_job_list)
  return coordinated_job_list_;
}

// int64 shutdown_barrier_timeout_in_ms = 7;
inline void CoordinationServiceConfig::clear_shutdown_barrier_timeout_in_ms() {
  shutdown_barrier_timeout_in_ms_ = int64_t{0};
}
inline ::PROTOBUF_NAMESPACE_ID::int64 CoordinationServiceConfig::_internal_shutdown_barrier_timeout_in_ms() const {
  return shutdown_barrier_timeout_in_ms_;
}
inline ::PROTOBUF_NAMESPACE_ID::int64 CoordinationServiceConfig::shutdown_barrier_timeout_in_ms() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.shutdown_barrier_timeout_in_ms)
  return _internal_shutdown_barrier_timeout_in_ms();
}
inline void CoordinationServiceConfig::_internal_set_shutdown_barrier_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value) {
  
  shutdown_barrier_timeout_in_ms_ = value;
}
inline void CoordinationServiceConfig::set_shutdown_barrier_timeout_in_ms(::PROTOBUF_NAMESPACE_ID::int64 value) {
  _internal_set_shutdown_barrier_timeout_in_ms(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.shutdown_barrier_timeout_in_ms)
}

// bool agent_destruction_without_shutdown = 8;
inline void CoordinationServiceConfig::clear_agent_destruction_without_shutdown() {
  agent_destruction_without_shutdown_ = false;
}
inline bool CoordinationServiceConfig::_internal_agent_destruction_without_shutdown() const {
  return agent_destruction_without_shutdown_;
}
inline bool CoordinationServiceConfig::agent_destruction_without_shutdown() const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.agent_destruction_without_shutdown)
  return _internal_agent_destruction_without_shutdown();
}
inline void CoordinationServiceConfig::_internal_set_agent_destruction_without_shutdown(bool value) {
  
  agent_destruction_without_shutdown_ = value;
}
inline void CoordinationServiceConfig::set_agent_destruction_without_shutdown(bool value) {
  _internal_set_agent_destruction_without_shutdown(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.agent_destruction_without_shutdown)
}

// repeated string recoverable_jobs = 9;
inline int CoordinationServiceConfig::_internal_recoverable_jobs_size() const {
  return recoverable_jobs_.size();
}
inline int CoordinationServiceConfig::recoverable_jobs_size() const {
  return _internal_recoverable_jobs_size();
}
inline void CoordinationServiceConfig::clear_recoverable_jobs() {
  recoverable_jobs_.Clear();
}
inline std::string* CoordinationServiceConfig::add_recoverable_jobs() {
  std::string* _s = _internal_add_recoverable_jobs();
  // @@protoc_insertion_point(field_add_mutable:tensorflow.CoordinationServiceConfig.recoverable_jobs)
  return _s;
}
inline const std::string& CoordinationServiceConfig::_internal_recoverable_jobs(int index) const {
  return recoverable_jobs_.Get(index);
}
inline const std::string& CoordinationServiceConfig::recoverable_jobs(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.CoordinationServiceConfig.recoverable_jobs)
  return _internal_recoverable_jobs(index);
}
inline std::string* CoordinationServiceConfig::mutable_recoverable_jobs(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.CoordinationServiceConfig.recoverable_jobs)
  return recoverable_jobs_.Mutable(index);
}
inline void CoordinationServiceConfig::set_recoverable_jobs(int index, const std::string& value) {
  recoverable_jobs_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline void CoordinationServiceConfig::set_recoverable_jobs(int index, std::string&& value) {
  recoverable_jobs_.Mutable(index)->assign(std::move(value));
  // @@protoc_insertion_point(field_set:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline void CoordinationServiceConfig::set_recoverable_jobs(int index, const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  recoverable_jobs_.Mutable(index)->assign(value);
  // @@protoc_insertion_point(field_set_char:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline void CoordinationServiceConfig::set_recoverable_jobs(int index, const char* value, size_t size) {
  recoverable_jobs_.Mutable(index)->assign(
    reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_set_pointer:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline std::string* CoordinationServiceConfig::_internal_add_recoverable_jobs() {
  return recoverable_jobs_.Add();
}
inline void CoordinationServiceConfig::add_recoverable_jobs(const std::string& value) {
  recoverable_jobs_.Add()->assign(value);
  // @@protoc_insertion_point(field_add:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline void CoordinationServiceConfig::add_recoverable_jobs(std::string&& value) {
  recoverable_jobs_.Add(std::move(value));
  // @@protoc_insertion_point(field_add:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline void CoordinationServiceConfig::add_recoverable_jobs(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  recoverable_jobs_.Add()->assign(value);
  // @@protoc_insertion_point(field_add_char:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline void CoordinationServiceConfig::add_recoverable_jobs(const char* value, size_t size) {
  recoverable_jobs_.Add()->assign(reinterpret_cast<const char*>(value), size);
  // @@protoc_insertion_point(field_add_pointer:tensorflow.CoordinationServiceConfig.recoverable_jobs)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>&
CoordinationServiceConfig::recoverable_jobs() const {
  // @@protoc_insertion_point(field_list:tensorflow.CoordinationServiceConfig.recoverable_jobs)
  return recoverable_jobs_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField<std::string>*
CoordinationServiceConfig::mutable_recoverable_jobs() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.CoordinationServiceConfig.recoverable_jobs)
  return &recoverable_jobs_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fcoordination_5fconfig_2eproto
