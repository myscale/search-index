/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::tfg::RegionAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::tfg::RegionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tfg::RegionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::tfg::RegionAttr>([&](auto t) {
      printer << ::mlir::tfg::RegionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace tfg {
namespace detail {
struct RegionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<DictionaryAttr, ArrayAttr, ArrayAttr>;
  RegionAttrStorage(DictionaryAttr attrs, ArrayAttr arg_attrs, ArrayAttr res_attrs) : attrs(attrs), arg_attrs(arg_attrs), res_attrs(res_attrs) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (attrs == std::get<0>(tblgenKey)) && (arg_attrs == std::get<1>(tblgenKey)) && (res_attrs == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static RegionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto attrs = std::get<0>(tblgenKey);
    auto arg_attrs = std::get<1>(tblgenKey);
    auto res_attrs = std::get<2>(tblgenKey);
    return new (allocator.allocate<RegionAttrStorage>()) RegionAttrStorage(attrs, arg_attrs, res_attrs);
  }

  DictionaryAttr attrs;
  ArrayAttr arg_attrs;
  ArrayAttr res_attrs;
};
} // namespace detail
RegionAttr RegionAttr::get(::mlir::MLIRContext *context, DictionaryAttr attrs, ArrayAttr arg_attrs, ArrayAttr res_attrs) {
  return Base::get(context, attrs, arg_attrs, res_attrs);
}

RegionAttr RegionAttr::get(DictionaryAttr attrs, ArrayAttr arg_attrs, ArrayAttr res_attrs) {
  return Base::get(attrs.getContext(), attrs, arg_attrs, res_attrs);
}

::mlir::Attribute RegionAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<DictionaryAttr> _result_attrs;
  ::mlir::FailureOr<ArrayAttr> _result_arg_attrs;
  ::mlir::FailureOr<ArrayAttr> _result_res_attrs;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'attrs'
  _result_attrs = ::mlir::FieldParser<DictionaryAttr>::parse(odsParser);
  if (::mlir::failed(_result_attrs)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TFGraph_RegionAttrs parameter 'attrs' which is to be a `DictionaryAttr`");
    return {};
  }

  // Parse variable 'arg_attrs'
  _result_arg_attrs = ::mlir::FieldParser<ArrayAttr>::parse(odsParser);
  if (::mlir::failed(_result_arg_attrs)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TFGraph_RegionAttrs parameter 'arg_attrs' which is to be a `ArrayAttr`");
    return {};
  }

  // Parse variable 'res_attrs'
  _result_res_attrs = ::mlir::FieldParser<ArrayAttr>::parse(odsParser);
  if (::mlir::failed(_result_res_attrs)) {
    odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TFGraph_RegionAttrs parameter 'res_attrs' which is to be a `ArrayAttr`");
    return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_attrs));
  assert(::mlir::succeeded(_result_arg_attrs));
  assert(::mlir::succeeded(_result_res_attrs));
  return RegionAttr::get(odsParser.getContext(),
      DictionaryAttr((*_result_attrs)),
      ArrayAttr((*_result_arg_attrs)),
      ArrayAttr((*_result_res_attrs)));
}

void RegionAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getAttrs());
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getArgAttrs());
  odsPrinter << ' ';
  odsPrinter.printStrippedAttrOrType(getResAttrs());
  odsPrinter << ">";
}

DictionaryAttr RegionAttr::getAttrs() const {
  return getImpl()->attrs;
}

ArrayAttr RegionAttr::getArgAttrs() const {
  return getImpl()->arg_attrs;
}

ArrayAttr RegionAttr::getResAttrs() const {
  return getImpl()->res_attrs;
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::RegionAttr)
namespace mlir {
namespace tfg {

/// Parse an attribute registered to this dialect.
::mlir::Attribute TFGraphDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                      ::mlir::Type type) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef attrTag;
  {
    ::mlir::Attribute attr;
    auto parseResult = generatedAttributeParser(parser, &attrTag, type, attr);
    if (parseResult.has_value())
      return attr;
  }
  
  parser.emitError(typeLoc) << "unknown attribute `"
      << attrTag << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print an attribute registered to this dialect.
void TFGraphDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace tfg
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

