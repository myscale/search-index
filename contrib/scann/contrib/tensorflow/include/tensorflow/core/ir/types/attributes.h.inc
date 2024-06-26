/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace tf_type {
class FullTypeAttr;
class FuncAttr;
class GpuDeviceMetadataAttr;
class PlaceholderAttr;
class ShapeAttr;
class VersionAttr;
class TensorProtoAttr;
namespace detail {
struct FullTypeAttrStorage;
} // namespace detail
class FullTypeAttr : public ::mlir::Attribute::AttrBase<FullTypeAttr, ::mlir::Attribute, detail::FullTypeAttrStorage> {
public:
  using Base::Base;
  static FullTypeAttr get(::mlir::MLIRContext *context, int32_t type_id, ::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr> args, Attribute attr);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"full_type"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int32_t getTypeId() const;
  ::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr> getArgs() const;
  Attribute getAttr() const;
};
namespace detail {
struct FuncAttrStorage;
} // namespace detail
class FuncAttr : public ::mlir::Attribute::AttrBase<FuncAttr, ::mlir::Attribute, detail::FuncAttrStorage, ::mlir::SubElementAttrInterface::Trait> {
public:
  using Base::Base;
  static FuncAttr get(::mlir::MLIRContext *context, SymbolRefAttr name, DictionaryAttr attrs);
  static FuncAttr get(::mlir::MLIRContext *context, StringRef name, DictionaryAttr attr);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"func"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  SymbolRefAttr getName() const;
  DictionaryAttr getAttrs() const;
  void walkImmediateSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn, llvm::function_ref<void(mlir::Type)> walkTypesFn) const;
  ::mlir::Attribute replaceImmediateSubElements(::llvm::ArrayRef<::mlir::Attribute> replAttrs, ::llvm::ArrayRef<::mlir::Type> replTypes) const;
};
namespace detail {
struct GpuDeviceMetadataAttrStorage;
} // namespace detail
class GpuDeviceMetadataAttr : public ::mlir::Attribute::AttrBase<GpuDeviceMetadataAttr, ::mlir::Attribute, detail::GpuDeviceMetadataAttrStorage> {
public:
  using Base::Base;
  static GpuDeviceMetadataAttr get(::mlir::MLIRContext *context, int32_t cc_major, int32_t cc_minor);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"gpu_device_metadata"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int32_t getCcMajor() const;
  int32_t getCcMinor() const;
};
namespace detail {
struct PlaceholderAttrStorage;
} // namespace detail
class PlaceholderAttr : public ::mlir::Attribute::AttrBase<PlaceholderAttr, ::mlir::Attribute, detail::PlaceholderAttrStorage> {
public:
  using Base::Base;
  static PlaceholderAttr get(::mlir::MLIRContext *context, ::llvm::StringRef value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"placeholder"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::StringRef getValue() const;
};
namespace detail {
struct ShapeAttrStorage;
} // namespace detail
class ShapeAttr : public ::mlir::Attribute::AttrBase<ShapeAttr, ::mlir::Attribute, detail::ShapeAttrStorage> {
public:
  using Base::Base;
  // Returns true if this shape is ranked and has only known dimensions size.
  bool hasStaticShape() const;

  // Returns true if this shape attribute has a statically known rank.
  bool hasRank() const;

  // Returns the rank. Aborts if unranked.
  int64_t getRank() const;

  // Returns the shape array if ranked, or None if unranked.
  llvm::Optional<ArrayRef<int64_t>> getValue() const;
  static ShapeAttr get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, bool unranked);
  static ShapeAttr get(::mlir::MLIRContext *context, llvm::Optional<ArrayRef<int64_t>> dimensions);
  static ShapeAttr get(::mlir::MLIRContext *context, ShapedType shaped_type);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"shape"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ::llvm::ArrayRef<int64_t> getShape() const;
  bool getUnranked() const;
};
namespace detail {
struct VersionAttrStorage;
} // namespace detail
class VersionAttr : public ::mlir::Attribute::AttrBase<VersionAttr, ::mlir::Attribute, detail::VersionAttrStorage> {
public:
  using Base::Base;
  static VersionAttr get(::mlir::MLIRContext *context, int32_t producer, int32_t minConsumer, ::llvm::ArrayRef<int32_t> badConsumers);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"version"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int32_t getProducer() const;
  int32_t getMinConsumer() const;
  ::llvm::ArrayRef<int32_t> getBadConsumers() const;
};
namespace detail {
struct TensorProtoAttrStorage;
} // namespace detail
class TensorProtoAttr : public ::mlir::Attribute::AttrBase<TensorProtoAttr, ::mlir::Attribute, detail::TensorProtoAttrStorage, ::mlir::ElementsAttr::Trait, ::mlir::TypedAttr::Trait> {
public:
  using Base::Base;
  using ValueType = StringRef;
  static TensorProtoAttr get(::mlir::MLIRContext *context, ShapedType type, ::llvm::StringRef value);
  static TensorProtoAttr get(ShapedType type, StringRef value);
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"tensor_proto"};
  }

  static ::mlir::Attribute parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  ShapedType getType() const;
  ::llvm::StringRef getValue() const;
};
} // namespace tf_type
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::FullTypeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::FuncAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::GpuDeviceMetadataAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::PlaceholderAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::ShapeAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::VersionAttr)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tf_type::TensorProtoAttr)

#endif  // GET_ATTRDEF_CLASSES

