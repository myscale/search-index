/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* AttrDef Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_ATTRDEF_LIST
#undef GET_ATTRDEF_LIST

::mlir::tf_type::FullTypeAttr,
::mlir::tf_type::FuncAttr,
::mlir::tf_type::GpuDeviceMetadataAttr,
::mlir::tf_type::PlaceholderAttr,
::mlir::tf_type::ShapeAttr,
::mlir::tf_type::VersionAttr,
::mlir::tf_type::TensorProtoAttr

#endif  // GET_ATTRDEF_LIST

#ifdef GET_ATTRDEF_CLASSES
#undef GET_ATTRDEF_CLASSES

static ::mlir::OptionalParseResult generatedAttributeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type type, ::mlir::Attribute &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::tf_type::FullTypeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::FullTypeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::tf_type::FuncAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::FuncAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::tf_type::GpuDeviceMetadataAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::GpuDeviceMetadataAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::tf_type::PlaceholderAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::PlaceholderAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::tf_type::ShapeAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::ShapeAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::tf_type::VersionAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::VersionAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Case(::mlir::tf_type::TensorProtoAttr::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tf_type::TensorProtoAttr::parse(parser, type);
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return llvm::None;
    });
}

static ::mlir::LogicalResult generatedAttributePrinter(::mlir::Attribute def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Attribute, ::mlir::LogicalResult>(def)    .Case<::mlir::tf_type::FullTypeAttr>([&](auto t) {
      printer << ::mlir::tf_type::FullTypeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::tf_type::FuncAttr>([&](auto t) {
      printer << ::mlir::tf_type::FuncAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::tf_type::GpuDeviceMetadataAttr>([&](auto t) {
      printer << ::mlir::tf_type::GpuDeviceMetadataAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::tf_type::PlaceholderAttr>([&](auto t) {
      printer << ::mlir::tf_type::PlaceholderAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::tf_type::ShapeAttr>([&](auto t) {
      printer << ::mlir::tf_type::ShapeAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::tf_type::VersionAttr>([&](auto t) {
      printer << ::mlir::tf_type::VersionAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Case<::mlir::tf_type::TensorProtoAttr>([&](auto t) {
      printer << ::mlir::tf_type::TensorProtoAttr::getMnemonic();
t.print(printer);
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace tf_type {
namespace detail {
struct FullTypeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int32_t, ::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr>, Attribute>;
  FullTypeAttrStorage(int32_t type_id, ::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr> args, Attribute attr) : type_id(type_id), args(args), attr(attr) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (type_id == std::get<0>(tblgenKey)) && (args == std::get<1>(tblgenKey)) && (attr == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static FullTypeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto type_id = std::get<0>(tblgenKey);
    auto args = std::get<1>(tblgenKey);
    auto attr = std::get<2>(tblgenKey);
    args = allocator.copyInto(args);
    return new (allocator.allocate<FullTypeAttrStorage>()) FullTypeAttrStorage(type_id, args, attr);
  }

  int32_t type_id;
  ::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr> args;
  Attribute attr;
};
} // namespace detail
FullTypeAttr FullTypeAttr::get(::mlir::MLIRContext *context, int32_t type_id, ::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr> args, Attribute attr) {
  return Base::get(context, type_id, args, attr);
}

int32_t FullTypeAttr::getTypeId() const {
  return getImpl()->type_id;
}

::llvm::ArrayRef<::mlir::tf_type::FullTypeAttr> FullTypeAttr::getArgs() const {
  return getImpl()->args;
}

Attribute FullTypeAttr::getAttr() const {
  return getImpl()->attr;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::FullTypeAttr)
namespace mlir {
namespace tf_type {
namespace detail {
struct FuncAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<SymbolRefAttr, DictionaryAttr>;
  FuncAttrStorage(SymbolRefAttr name, DictionaryAttr attrs) : name(name), attrs(attrs) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (name == std::get<0>(tblgenKey)) && (attrs == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static FuncAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto name = std::get<0>(tblgenKey);
    auto attrs = std::get<1>(tblgenKey);
    return new (allocator.allocate<FuncAttrStorage>()) FuncAttrStorage(name, attrs);
  }

  SymbolRefAttr name;
  DictionaryAttr attrs;
};
} // namespace detail
FuncAttr FuncAttr::get(::mlir::MLIRContext *context, SymbolRefAttr name, DictionaryAttr attrs) {
  return Base::get(context, name, attrs);
}

FuncAttr FuncAttr::get(::mlir::MLIRContext *context, StringRef name, DictionaryAttr attr) {
  return Base::get(context, SymbolRefAttr::get(context, name), attr);
}

SymbolRefAttr FuncAttr::getName() const {
  return getImpl()->name;
}

DictionaryAttr FuncAttr::getAttrs() const {
  return getImpl()->attrs;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::FuncAttr)
namespace mlir {
namespace tf_type {
namespace detail {
struct GpuDeviceMetadataAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int32_t, int32_t>;
  GpuDeviceMetadataAttrStorage(int32_t cc_major, int32_t cc_minor) : cc_major(cc_major), cc_minor(cc_minor) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (cc_major == std::get<0>(tblgenKey)) && (cc_minor == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static GpuDeviceMetadataAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto cc_major = std::get<0>(tblgenKey);
    auto cc_minor = std::get<1>(tblgenKey);
    return new (allocator.allocate<GpuDeviceMetadataAttrStorage>()) GpuDeviceMetadataAttrStorage(cc_major, cc_minor);
  }

  int32_t cc_major;
  int32_t cc_minor;
};
} // namespace detail
GpuDeviceMetadataAttr GpuDeviceMetadataAttr::get(::mlir::MLIRContext *context, int32_t cc_major, int32_t cc_minor) {
  return Base::get(context, cc_major, cc_minor);
}

::mlir::Attribute GpuDeviceMetadataAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  ::mlir::Builder odsBuilder(odsParser.getContext());
  ::llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  ::mlir::FailureOr<int32_t> _result_cc_major;
  ::mlir::FailureOr<int32_t> _result_cc_minor;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};
  // Parse parameter struct
  bool _seen_cc_major = false;
  bool _seen_cc_minor = false;
  {
    const auto _loop_body = [&](::llvm::StringRef _paramKey) -> bool {
      // Parse literal '='
      if (odsParser.parseEqual()) return {};
      if (!_seen_cc_major && _paramKey == "cc_major") {
        _seen_cc_major = true;

        // Parse variable 'cc_major'
        _result_cc_major = ::mlir::FieldParser<int32_t>::parse(odsParser);
        if (::mlir::failed(_result_cc_major)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TFType_GpuDeviceMetadata parameter 'cc_major' which is to be a `int32_t`");
          return {};
        }
      } else if (!_seen_cc_minor && _paramKey == "cc_minor") {
        _seen_cc_minor = true;

        // Parse variable 'cc_minor'
        _result_cc_minor = ::mlir::FieldParser<int32_t>::parse(odsParser);
        if (::mlir::failed(_result_cc_minor)) {
          odsParser.emitError(odsParser.getCurrentLocation(), "failed to parse TFType_GpuDeviceMetadata parameter 'cc_minor' which is to be a `int32_t`");
          return {};
        }
      } else {
        odsParser.emitError(odsParser.getCurrentLocation(), "duplicate or unknown struct parameter name: ") << _paramKey;
        return {};
      }
      return true;
    };
    for (unsigned odsStructIndex = 0; odsStructIndex < 2; ++odsStructIndex) {
      ::llvm::StringRef _paramKey;
      if (odsParser.parseKeyword(&_paramKey)) {
        odsParser.emitError(odsParser.getCurrentLocation(),
                           "expected a parameter name in struct");
        return {};
      }
      if (!_loop_body(_paramKey)) return {};
      if ((odsStructIndex != 2 - 1) && odsParser.parseComma())
        return {};
    }
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(::mlir::succeeded(_result_cc_major));
  assert(::mlir::succeeded(_result_cc_minor));
  return GpuDeviceMetadataAttr::get(odsParser.getContext(),
      int32_t((*_result_cc_major)),
      int32_t((*_result_cc_minor)));
}

void GpuDeviceMetadataAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  {
    bool _firstPrinted = true;
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "cc_major = ";
    odsPrinter.printStrippedAttrOrType(getCcMajor());
    if (!_firstPrinted) odsPrinter << ", ";
    _firstPrinted = false;
    odsPrinter << "cc_minor = ";
    odsPrinter.printStrippedAttrOrType(getCcMinor());
  }
  odsPrinter << ">";
}

int32_t GpuDeviceMetadataAttr::getCcMajor() const {
  return getImpl()->cc_major;
}

int32_t GpuDeviceMetadataAttr::getCcMinor() const {
  return getImpl()->cc_minor;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::GpuDeviceMetadataAttr)
namespace mlir {
namespace tf_type {
namespace detail {
struct PlaceholderAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::StringRef>;
  PlaceholderAttrStorage(::llvm::StringRef value) : value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (value == std::get<0>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey));
  }

  static PlaceholderAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto value = std::get<0>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<PlaceholderAttrStorage>()) PlaceholderAttrStorage(value);
  }

  ::llvm::StringRef value;
};
} // namespace detail
PlaceholderAttr PlaceholderAttr::get(::mlir::MLIRContext *context, ::llvm::StringRef value) {
  return Base::get(context, value);
}

::llvm::StringRef PlaceholderAttr::getValue() const {
  return getImpl()->value;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::PlaceholderAttr)
namespace mlir {
namespace tf_type {
namespace detail {
struct ShapeAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<::llvm::ArrayRef<int64_t>, bool>;
  ShapeAttrStorage(::llvm::ArrayRef<int64_t> shape, bool unranked) : shape(shape), unranked(unranked) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (shape == std::get<0>(tblgenKey)) && (unranked == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static ShapeAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto shape = std::get<0>(tblgenKey);
    auto unranked = std::get<1>(tblgenKey);
    shape = allocator.copyInto(shape);
    return new (allocator.allocate<ShapeAttrStorage>()) ShapeAttrStorage(shape, unranked);
  }

  ::llvm::ArrayRef<int64_t> shape;
  bool unranked;
};
} // namespace detail
ShapeAttr ShapeAttr::get(::mlir::MLIRContext *context, ::llvm::ArrayRef<int64_t> shape, bool unranked) {
  return Base::get(context, shape, unranked);
}

::llvm::ArrayRef<int64_t> ShapeAttr::getShape() const {
  return getImpl()->shape;
}

bool ShapeAttr::getUnranked() const {
  return getImpl()->unranked;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::ShapeAttr)
namespace mlir {
namespace tf_type {
namespace detail {
struct VersionAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<int32_t, int32_t, ::llvm::ArrayRef<int32_t>>;
  VersionAttrStorage(int32_t producer, int32_t minConsumer, ::llvm::ArrayRef<int32_t> badConsumers) : producer(producer), minConsumer(minConsumer), badConsumers(badConsumers) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (producer == std::get<0>(tblgenKey)) && (minConsumer == std::get<1>(tblgenKey)) && (badConsumers == std::get<2>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey), std::get<2>(tblgenKey));
  }

  static VersionAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto producer = std::get<0>(tblgenKey);
    auto minConsumer = std::get<1>(tblgenKey);
    auto badConsumers = std::get<2>(tblgenKey);
    badConsumers = allocator.copyInto(badConsumers);
    return new (allocator.allocate<VersionAttrStorage>()) VersionAttrStorage(producer, minConsumer, badConsumers);
  }

  int32_t producer;
  int32_t minConsumer;
  ::llvm::ArrayRef<int32_t> badConsumers;
};
} // namespace detail
VersionAttr VersionAttr::get(::mlir::MLIRContext *context, int32_t producer, int32_t minConsumer, ::llvm::ArrayRef<int32_t> badConsumers) {
  return Base::get(context, producer, minConsumer, badConsumers);
}

int32_t VersionAttr::getProducer() const {
  return getImpl()->producer;
}

int32_t VersionAttr::getMinConsumer() const {
  return getImpl()->minConsumer;
}

::llvm::ArrayRef<int32_t> VersionAttr::getBadConsumers() const {
  return getImpl()->badConsumers;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::VersionAttr)
namespace mlir {
namespace tf_type {
namespace detail {
struct TensorProtoAttrStorage : public ::mlir::AttributeStorage {
  using KeyTy = std::tuple<ShapedType, ::llvm::StringRef>;
  TensorProtoAttrStorage(ShapedType type, ::llvm::StringRef value) : type(type), value(value) {}

  bool operator==(const KeyTy &tblgenKey) const {
    return (type == std::get<0>(tblgenKey)) && (value == std::get<1>(tblgenKey));
  }

  static ::llvm::hash_code hashKey(const KeyTy &tblgenKey) {
    return ::llvm::hash_combine(std::get<0>(tblgenKey), std::get<1>(tblgenKey));
  }

  static TensorProtoAttrStorage *construct(::mlir::AttributeStorageAllocator &allocator, const KeyTy &tblgenKey) {
    auto type = std::get<0>(tblgenKey);
    auto value = std::get<1>(tblgenKey);
    value = allocator.copyInto(value);
    return new (allocator.allocate<TensorProtoAttrStorage>()) TensorProtoAttrStorage(type, value);
  }

  ShapedType type;
  ::llvm::StringRef value;
};
} // namespace detail
TensorProtoAttr TensorProtoAttr::get(::mlir::MLIRContext *context, ShapedType type, ::llvm::StringRef value) {
  return Base::get(context, type, value);
}

TensorProtoAttr TensorProtoAttr::get(ShapedType type, StringRef value) {
  return Base::get(type.getContext(), type, value);
}

ShapedType TensorProtoAttr::getType() const {
  return getImpl()->type;
}

::llvm::StringRef TensorProtoAttr::getValue() const {
  return getImpl()->value;
}

} // namespace tf_type
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tf_type::TensorProtoAttr)
namespace mlir {
namespace tf_type {

/// Parse an attribute registered to this dialect.
::mlir::Attribute TFTypeDialect::parseAttribute(::mlir::DialectAsmParser &parser,
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
void TFTypeDialect::printAttribute(::mlir::Attribute attr,
                         ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedAttributePrinter(attr, printer)))
    return;
  
}
} // namespace tf_type
} // namespace mlir

#endif  // GET_ATTRDEF_CLASSES

