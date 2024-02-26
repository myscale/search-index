/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Definitions                                                   *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace tf_type {
::llvm::StringRef stringifyFullTypeId(FullTypeId val) {
  switch (val) {
    case FullTypeId::TFT_UNSET: return "unset";
    case FullTypeId::TFT_VAR: return "var";
    case FullTypeId::TFT_ANY: return "any";
    case FullTypeId::TFT_PRODUCT: return "product";
    case FullTypeId::TFT_NAMED: return "named";
    case FullTypeId::TFT_FOR_EACH: return "for_each";
    case FullTypeId::TFT_CALLABLE: return "callable";
    case FullTypeId::TFT_TENSOR: return "tensor";
    case FullTypeId::TFT_ARRAY: return "array";
    case FullTypeId::TFT_OPTIONAL: return "optional";
    case FullTypeId::TFT_LITERAL: return "literal";
    case FullTypeId::TFT_ENCODED: return "encoded";
    case FullTypeId::TFT_BOOL: return "bool";
    case FullTypeId::TFT_UINT8: return "uint8";
    case FullTypeId::TFT_UINT16: return "uint16";
    case FullTypeId::TFT_UINT32: return "uint32";
    case FullTypeId::TFT_UINT64: return "uint64";
    case FullTypeId::TFT_INT8: return "int8";
    case FullTypeId::TFT_INT16: return "int16";
    case FullTypeId::TFT_INT32: return "int32";
    case FullTypeId::TFT_INT64: return "int64";
    case FullTypeId::TFT_HALF: return "half";
    case FullTypeId::TFT_FLOAT: return "float";
    case FullTypeId::TFT_DOUBLE: return "double";
    case FullTypeId::TFT_BFLOAT16: return "bfloat16";
    case FullTypeId::TFT_COMPLEX64: return "complex64";
    case FullTypeId::TFT_COMPLEX128: return "complex128";
    case FullTypeId::TFT_STRING: return "string";
    case FullTypeId::TFT_DATASET: return "dataset";
    case FullTypeId::TFT_RAGGED: return "ragged";
    case FullTypeId::TFT_ITERATOR: return "iterator";
    case FullTypeId::TFT_MUTEX_LOCK: return "mutex_lock";
    case FullTypeId::TFT_LEGACY_VARIANT: return "legacy_variant";
  }
  return "";
}

::llvm::Optional<FullTypeId> symbolizeFullTypeId(::llvm::StringRef str) {
  return ::llvm::StringSwitch<::llvm::Optional<FullTypeId>>(str)
      .Case("unset", FullTypeId::TFT_UNSET)
      .Case("var", FullTypeId::TFT_VAR)
      .Case("any", FullTypeId::TFT_ANY)
      .Case("product", FullTypeId::TFT_PRODUCT)
      .Case("named", FullTypeId::TFT_NAMED)
      .Case("for_each", FullTypeId::TFT_FOR_EACH)
      .Case("callable", FullTypeId::TFT_CALLABLE)
      .Case("tensor", FullTypeId::TFT_TENSOR)
      .Case("array", FullTypeId::TFT_ARRAY)
      .Case("optional", FullTypeId::TFT_OPTIONAL)
      .Case("literal", FullTypeId::TFT_LITERAL)
      .Case("encoded", FullTypeId::TFT_ENCODED)
      .Case("bool", FullTypeId::TFT_BOOL)
      .Case("uint8", FullTypeId::TFT_UINT8)
      .Case("uint16", FullTypeId::TFT_UINT16)
      .Case("uint32", FullTypeId::TFT_UINT32)
      .Case("uint64", FullTypeId::TFT_UINT64)
      .Case("int8", FullTypeId::TFT_INT8)
      .Case("int16", FullTypeId::TFT_INT16)
      .Case("int32", FullTypeId::TFT_INT32)
      .Case("int64", FullTypeId::TFT_INT64)
      .Case("half", FullTypeId::TFT_HALF)
      .Case("float", FullTypeId::TFT_FLOAT)
      .Case("double", FullTypeId::TFT_DOUBLE)
      .Case("bfloat16", FullTypeId::TFT_BFLOAT16)
      .Case("complex64", FullTypeId::TFT_COMPLEX64)
      .Case("complex128", FullTypeId::TFT_COMPLEX128)
      .Case("string", FullTypeId::TFT_STRING)
      .Case("dataset", FullTypeId::TFT_DATASET)
      .Case("ragged", FullTypeId::TFT_RAGGED)
      .Case("iterator", FullTypeId::TFT_ITERATOR)
      .Case("mutex_lock", FullTypeId::TFT_MUTEX_LOCK)
      .Case("legacy_variant", FullTypeId::TFT_LEGACY_VARIANT)
      .Default(::llvm::None);
}
::llvm::Optional<FullTypeId> symbolizeFullTypeId(uint32_t value) {
  switch (value) {
  case 0: return FullTypeId::TFT_UNSET;
  case 1: return FullTypeId::TFT_VAR;
  case 2: return FullTypeId::TFT_ANY;
  case 3: return FullTypeId::TFT_PRODUCT;
  case 4: return FullTypeId::TFT_NAMED;
  case 20: return FullTypeId::TFT_FOR_EACH;
  case 100: return FullTypeId::TFT_CALLABLE;
  case 1000: return FullTypeId::TFT_TENSOR;
  case 1001: return FullTypeId::TFT_ARRAY;
  case 1002: return FullTypeId::TFT_OPTIONAL;
  case 1003: return FullTypeId::TFT_LITERAL;
  case 1004: return FullTypeId::TFT_ENCODED;
  case 200: return FullTypeId::TFT_BOOL;
  case 201: return FullTypeId::TFT_UINT8;
  case 202: return FullTypeId::TFT_UINT16;
  case 203: return FullTypeId::TFT_UINT32;
  case 204: return FullTypeId::TFT_UINT64;
  case 205: return FullTypeId::TFT_INT8;
  case 206: return FullTypeId::TFT_INT16;
  case 207: return FullTypeId::TFT_INT32;
  case 208: return FullTypeId::TFT_INT64;
  case 209: return FullTypeId::TFT_HALF;
  case 210: return FullTypeId::TFT_FLOAT;
  case 211: return FullTypeId::TFT_DOUBLE;
  case 215: return FullTypeId::TFT_BFLOAT16;
  case 212: return FullTypeId::TFT_COMPLEX64;
  case 213: return FullTypeId::TFT_COMPLEX128;
  case 214: return FullTypeId::TFT_STRING;
  case 10102: return FullTypeId::TFT_DATASET;
  case 10103: return FullTypeId::TFT_RAGGED;
  case 10104: return FullTypeId::TFT_ITERATOR;
  case 10202: return FullTypeId::TFT_MUTEX_LOCK;
  case 10203: return FullTypeId::TFT_LEGACY_VARIANT;
  default: return ::llvm::None;
  }
}

} // namespace tf_type
} // namespace mlir

