/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tf_type {
// allowed 32-bit signless integer cases: 0, 1, 2, 3, 4, 20, 100, 1000, 1001, 1002, 1003, 1004, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 215, 212, 213, 214, 10102, 10103, 10104, 10202, 10203
enum class FullTypeId : uint32_t {
  TFT_UNSET = 0,
  TFT_VAR = 1,
  TFT_ANY = 2,
  TFT_PRODUCT = 3,
  TFT_NAMED = 4,
  TFT_FOR_EACH = 20,
  TFT_CALLABLE = 100,
  TFT_TENSOR = 1000,
  TFT_ARRAY = 1001,
  TFT_OPTIONAL = 1002,
  TFT_LITERAL = 1003,
  TFT_ENCODED = 1004,
  TFT_BOOL = 200,
  TFT_UINT8 = 201,
  TFT_UINT16 = 202,
  TFT_UINT32 = 203,
  TFT_UINT64 = 204,
  TFT_INT8 = 205,
  TFT_INT16 = 206,
  TFT_INT32 = 207,
  TFT_INT64 = 208,
  TFT_HALF = 209,
  TFT_FLOAT = 210,
  TFT_DOUBLE = 211,
  TFT_BFLOAT16 = 215,
  TFT_COMPLEX64 = 212,
  TFT_COMPLEX128 = 213,
  TFT_STRING = 214,
  TFT_DATASET = 10102,
  TFT_RAGGED = 10103,
  TFT_ITERATOR = 10104,
  TFT_MUTEX_LOCK = 10202,
  TFT_LEGACY_VARIANT = 10203,
};

::llvm::Optional<FullTypeId> symbolizeFullTypeId(uint32_t);
::llvm::StringRef stringifyFullTypeId(FullTypeId);
::llvm::Optional<FullTypeId> symbolizeFullTypeId(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForFullTypeId() {
  return 10203;
}


inline ::llvm::StringRef stringifyEnum(FullTypeId enumValue) {
  return stringifyFullTypeId(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<FullTypeId> symbolizeEnum<FullTypeId>(::llvm::StringRef str) {
  return symbolizeFullTypeId(str);
}
} // namespace tf_type
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::tf_type::FullTypeId> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::tf_type::FullTypeId getEmptyKey() {
    return static_cast<::mlir::tf_type::FullTypeId>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::tf_type::FullTypeId getTombstoneKey() {
    return static_cast<::mlir::tf_type::FullTypeId>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::tf_type::FullTypeId &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::tf_type::FullTypeId &lhs, const ::mlir::tf_type::FullTypeId &rhs) {
    return lhs == rhs;
  }
};
}

