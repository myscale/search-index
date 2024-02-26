// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: scann/scann_ops/scann_assets.proto

#include "scann/scann_ops/scann_assets.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG
namespace research_scann {
constexpr ScannAsset::ScannAsset(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : asset_path_(&::PROTOBUF_NAMESPACE_ID::internal::fixed_address_empty_string)
  , asset_type_(0)
{}
struct ScannAssetDefaultTypeInternal {
  constexpr ScannAssetDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ScannAssetDefaultTypeInternal() {}
  union {
    ScannAsset _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ScannAssetDefaultTypeInternal _ScannAsset_default_instance_;
constexpr ScannAssets::ScannAssets(
  ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized)
  : assets_()
  , trained_on_the_fly_(true){}
struct ScannAssetsDefaultTypeInternal {
  constexpr ScannAssetsDefaultTypeInternal()
    : _instance(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized{}) {}
  ~ScannAssetsDefaultTypeInternal() {}
  union {
    ScannAssets _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT ScannAssetsDefaultTypeInternal _ScannAssets_default_instance_;
}  // namespace research_scann
static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_scann_2fscann_5fops_2fscann_5fassets_2eproto[2];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_scann_2fscann_5fops_2fscann_5fassets_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_scann_2fscann_5fops_2fscann_5fassets_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_scann_2fscann_5fops_2fscann_5fassets_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAsset, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAsset, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAsset, asset_type_),
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAsset, asset_path_),
  1,
  0,
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAssets, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAssets, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAssets, assets_),
  PROTOBUF_FIELD_OFFSET(::research_scann::ScannAssets, trained_on_the_fly_),
  ~0u,
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::research_scann::ScannAsset)},
  { 9, 16, sizeof(::research_scann::ScannAssets)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::research_scann::_ScannAsset_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::research_scann::_ScannAssets_default_instance_),
};

const char descriptor_table_protodef_scann_2fscann_5fops_2fscann_5fassets_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\"scann/scann_ops/scann_assets.proto\022\016re"
  "search_scann\"\242\003\n\nScannAsset\0228\n\nasset_typ"
  "e\030\001 \001(\0162$.research_scann.ScannAsset.Asse"
  "tType\022\022\n\nasset_path\030\002 \001(\t\"\305\002\n\tAssetType\022"
  "\024\n\020UNSPECIFIED_TYPE\020\000\022\013\n\007DATASET\020\001\022\020\n\014IN"
  "T8_DATASET\020\002\022\016\n\nAH_DATASET\020\003\022\020\n\014TOKENIZA"
  "TION\020\004\022\037\n\033REORDERING_INT8_MULTIPLIERS\020\005\022"
  " \n\034BRUTE_FORCE_INT8_MULTIPLIERS\020\006\022\016\n\nAH_"
  "CENTERS\020\007\022\017\n\013PARTITIONER\020\010\022\017\n\013DATASET_NP"
  "Y\020\t\022\024\n\020INT8_DATASET_NPY\020\n\022\022\n\016AH_DATASET_"
  "NPY\020\013\022\024\n\020TOKENIZATION_NPY\020\014\022\030\n\024INT8_MULT"
  "IPLIERS_NPY\020\r\022\022\n\016INT8_NORMS_NPY\020\016\"[\n\013Sca"
  "nnAssets\022*\n\006assets\030\001 \003(\0132\032.research_scan"
  "n.ScannAsset\022 \n\022trained_on_the_fly\030\002 \001(\010"
  ":\004true"
  ;
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto = {
  false, false, 566, descriptor_table_protodef_scann_2fscann_5fops_2fscann_5fassets_2eproto, "scann/scann_ops/scann_assets.proto", 
  &descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_once, nullptr, 0, 2,
  schemas, file_default_instances, TableStruct_scann_2fscann_5fops_2fscann_5fassets_2eproto::offsets,
  file_level_metadata_scann_2fscann_5fops_2fscann_5fassets_2eproto, file_level_enum_descriptors_scann_2fscann_5fops_2fscann_5fassets_2eproto, file_level_service_descriptors_scann_2fscann_5fops_2fscann_5fassets_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable* descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_getter() {
  return &descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY static ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptorsRunner dynamic_init_dummy_scann_2fscann_5fops_2fscann_5fassets_2eproto(&descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto);
namespace research_scann {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* ScannAsset_AssetType_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto);
  return file_level_enum_descriptors_scann_2fscann_5fops_2fscann_5fassets_2eproto[0];
}
bool ScannAsset_AssetType_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr ScannAsset_AssetType ScannAsset::UNSPECIFIED_TYPE;
constexpr ScannAsset_AssetType ScannAsset::DATASET;
constexpr ScannAsset_AssetType ScannAsset::INT8_DATASET;
constexpr ScannAsset_AssetType ScannAsset::AH_DATASET;
constexpr ScannAsset_AssetType ScannAsset::TOKENIZATION;
constexpr ScannAsset_AssetType ScannAsset::REORDERING_INT8_MULTIPLIERS;
constexpr ScannAsset_AssetType ScannAsset::BRUTE_FORCE_INT8_MULTIPLIERS;
constexpr ScannAsset_AssetType ScannAsset::AH_CENTERS;
constexpr ScannAsset_AssetType ScannAsset::PARTITIONER;
constexpr ScannAsset_AssetType ScannAsset::DATASET_NPY;
constexpr ScannAsset_AssetType ScannAsset::INT8_DATASET_NPY;
constexpr ScannAsset_AssetType ScannAsset::AH_DATASET_NPY;
constexpr ScannAsset_AssetType ScannAsset::TOKENIZATION_NPY;
constexpr ScannAsset_AssetType ScannAsset::INT8_MULTIPLIERS_NPY;
constexpr ScannAsset_AssetType ScannAsset::INT8_NORMS_NPY;
constexpr ScannAsset_AssetType ScannAsset::AssetType_MIN;
constexpr ScannAsset_AssetType ScannAsset::AssetType_MAX;
constexpr int ScannAsset::AssetType_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

class ScannAsset::_Internal {
 public:
  using HasBits = decltype(std::declval<ScannAsset>()._has_bits_);
  static void set_has_asset_type(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static void set_has_asset_path(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

ScannAsset::ScannAsset(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:research_scann.ScannAsset)
}
ScannAsset::ScannAsset(const ScannAsset& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  asset_path_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from._internal_has_asset_path()) {
    asset_path_.Set(::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::EmptyDefault{}, from._internal_asset_path(), 
      GetArenaForAllocation());
  }
  asset_type_ = from.asset_type_;
  // @@protoc_insertion_point(copy_constructor:research_scann.ScannAsset)
}

inline void ScannAsset::SharedCtor() {
asset_path_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
asset_type_ = 0;
}

ScannAsset::~ScannAsset() {
  // @@protoc_insertion_point(destructor:research_scann.ScannAsset)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void ScannAsset::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  asset_path_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void ScannAsset::ArenaDtor(void* object) {
  ScannAsset* _this = reinterpret_cast< ScannAsset* >(object);
  (void)_this;
}
void ScannAsset::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ScannAsset::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void ScannAsset::Clear() {
// @@protoc_insertion_point(message_clear_start:research_scann.ScannAsset)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    asset_path_.ClearNonDefaultToEmpty();
  }
  asset_type_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ScannAsset::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // optional .research_scann.ScannAsset.AssetType asset_type = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          if (PROTOBUF_PREDICT_TRUE(::research_scann::ScannAsset_AssetType_IsValid(val))) {
            _internal_set_asset_type(static_cast<::research_scann::ScannAsset_AssetType>(val));
          } else {
            ::PROTOBUF_NAMESPACE_ID::internal::WriteVarint(1, val, mutable_unknown_fields());
          }
        } else goto handle_unusual;
        continue;
      // optional string asset_path = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          auto str = _internal_mutable_asset_path();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          #ifndef NDEBUG
          ::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "research_scann.ScannAsset.asset_path");
          #endif  // !NDEBUG
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag == 0) || ((tag & 7) == 4)) {
          CHK_(ptr);
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ScannAsset::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:research_scann.ScannAsset)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .research_scann.ScannAsset.AssetType asset_type = 1;
  if (cached_has_bits & 0x00000002u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      1, this->_internal_asset_type(), target);
  }

  // optional string asset_path = 2;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->_internal_asset_path().data(), static_cast<int>(this->_internal_asset_path().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "research_scann.ScannAsset.asset_path");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_asset_path(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:research_scann.ScannAsset)
  return target;
}

size_t ScannAsset::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:research_scann.ScannAsset)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional string asset_path = 2;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_asset_path());
    }

    // optional .research_scann.ScannAsset.AssetType asset_type = 1;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_asset_type());
    }

  }
  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ScannAsset::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    ScannAsset::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ScannAsset::GetClassData() const { return &_class_data_; }

void ScannAsset::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to,
                      const ::PROTOBUF_NAMESPACE_ID::Message&from) {
  static_cast<ScannAsset *>(to)->MergeFrom(
      static_cast<const ScannAsset &>(from));
}


void ScannAsset::MergeFrom(const ScannAsset& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:research_scann.ScannAsset)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      _internal_set_asset_path(from._internal_asset_path());
    }
    if (cached_has_bits & 0x00000002u) {
      asset_type_ = from.asset_type_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ScannAsset::CopyFrom(const ScannAsset& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:research_scann.ScannAsset)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ScannAsset::IsInitialized() const {
  return true;
}

void ScannAsset::InternalSwap(ScannAsset* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      &asset_path_, GetArenaForAllocation(),
      &other->asset_path_, other->GetArenaForAllocation()
  );
  swap(asset_type_, other->asset_type_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ScannAsset::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_getter, &descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_once,
      file_level_metadata_scann_2fscann_5fops_2fscann_5fassets_2eproto[0]);
}

// ===================================================================

class ScannAssets::_Internal {
 public:
  using HasBits = decltype(std::declval<ScannAssets>()._has_bits_);
  static void set_has_trained_on_the_fly(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

ScannAssets::ScannAssets(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned),
  assets_(arena) {
  SharedCtor();
  if (!is_message_owned) {
    RegisterArenaDtor(arena);
  }
  // @@protoc_insertion_point(arena_constructor:research_scann.ScannAssets)
}
ScannAssets::ScannAssets(const ScannAssets& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _has_bits_(from._has_bits_),
      assets_(from.assets_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  trained_on_the_fly_ = from.trained_on_the_fly_;
  // @@protoc_insertion_point(copy_constructor:research_scann.ScannAssets)
}

inline void ScannAssets::SharedCtor() {
trained_on_the_fly_ = true;
}

ScannAssets::~ScannAssets() {
  // @@protoc_insertion_point(destructor:research_scann.ScannAssets)
  if (GetArenaForAllocation() != nullptr) return;
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

inline void ScannAssets::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void ScannAssets::ArenaDtor(void* object) {
  ScannAssets* _this = reinterpret_cast< ScannAssets* >(object);
  (void)_this;
}
void ScannAssets::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void ScannAssets::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}

void ScannAssets::Clear() {
// @@protoc_insertion_point(message_clear_start:research_scann.ScannAssets)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  assets_.Clear();
  trained_on_the_fly_ = true;
  _has_bits_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ScannAssets::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .research_scann.ScannAsset assets = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_assets(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else goto handle_unusual;
        continue;
      // optional bool trained_on_the_fly = 2 [default = true];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          _Internal::set_has_trained_on_the_fly(&has_bits);
          trained_on_the_fly_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag == 0) || ((tag & 7) == 4)) {
          CHK_(ptr);
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* ScannAssets::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:research_scann.ScannAssets)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .research_scann.ScannAsset assets = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_assets_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, this->_internal_assets(i), target, stream);
  }

  cached_has_bits = _has_bits_[0];
  // optional bool trained_on_the_fly = 2 [default = true];
  if (cached_has_bits & 0x00000001u) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(2, this->_internal_trained_on_the_fly(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:research_scann.ScannAssets)
  return target;
}

size_t ScannAssets::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:research_scann.ScannAssets)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .research_scann.ScannAsset assets = 1;
  total_size += 1UL * this->_internal_assets_size();
  for (const auto& msg : this->assets_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // optional bool trained_on_the_fly = 2 [default = true];
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += 1 + 1;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ScannAssets::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSizeCheck,
    ScannAssets::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ScannAssets::GetClassData() const { return &_class_data_; }

void ScannAssets::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message*to,
                      const ::PROTOBUF_NAMESPACE_ID::Message&from) {
  static_cast<ScannAssets *>(to)->MergeFrom(
      static_cast<const ScannAssets &>(from));
}


void ScannAssets::MergeFrom(const ScannAssets& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:research_scann.ScannAssets)
  GOOGLE_DCHECK_NE(&from, this);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  assets_.MergeFrom(from.assets_);
  if (from._internal_has_trained_on_the_fly()) {
    _internal_set_trained_on_the_fly(from._internal_trained_on_the_fly());
  }
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ScannAssets::CopyFrom(const ScannAssets& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:research_scann.ScannAssets)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ScannAssets::IsInitialized() const {
  return true;
}

void ScannAssets::InternalSwap(ScannAssets* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  assets_.InternalSwap(&other->assets_);
  swap(trained_on_the_fly_, other->trained_on_the_fly_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ScannAssets::GetMetadata() const {
  return ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(
      &descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_getter, &descriptor_table_scann_2fscann_5fops_2fscann_5fassets_2eproto_once,
      file_level_metadata_scann_2fscann_5fops_2fscann_5fassets_2eproto[1]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace research_scann
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::research_scann::ScannAsset* Arena::CreateMaybeMessage< ::research_scann::ScannAsset >(Arena* arena) {
  return Arena::CreateMessageInternal< ::research_scann::ScannAsset >(arena);
}
template<> PROTOBUF_NOINLINE ::research_scann::ScannAssets* Arena::CreateMaybeMessage< ::research_scann::ScannAssets >(Arena* arena) {
  return Arena::CreateMessageInternal< ::research_scann::ScannAssets >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
