/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::tfg::CaseOp,
::mlir::tfg::CaseRegionOp,
::mlir::tfg::ConditionOp,
::mlir::tfg::ForOp,
::mlir::tfg::ForRegionOp,
::mlir::tfg::GetResultOp,
::mlir::tfg::GraphFuncOp,
::mlir::tfg::GraphOp,
::mlir::tfg::IfOp,
::mlir::tfg::IfRegionOp,
::mlir::tfg::ReturnOp,
::mlir::tfg::StatefulCaseOp,
::mlir::tfg::StatefulCaseRegionOp,
::mlir::tfg::StatefulIfOp,
::mlir::tfg::StatefulIfRegionOp,
::mlir::tfg::StatefulWhileOp,
::mlir::tfg::StatefulWhileRegionOp,
::mlir::tfg::StatelessCaseOp,
::mlir::tfg::StatelessCaseRegionOp,
::mlir::tfg::StatelessIfOp,
::mlir::tfg::StatelessIfRegionOp,
::mlir::tfg::StatelessWhileOp,
::mlir::tfg::StatelessWhileRegionOp,
::mlir::tfg::WhileOp,
::mlir::tfg::WhileRegionOp,
::mlir::tfg::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tfg {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((type.isa<::mlir::tf_type::OpaqueTensorType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be  or opaque tensor type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((type.isa<::mlir::tf_type::OpaqueTensorType>()))) || ((type.isa<::mlir::tf_type::ControlType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any tensor or control type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((type.isa<::mlir::tf_type::OpaqueTensorType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be  or opaque tensor type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::tf_type::ControlType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 1-bit signless integer values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::tf_type::OpaqueTensorType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(1)); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((type.isa<::mlir::tf_type::OpaqueTensorType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be  or opaque tensor type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_ops8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::FuncAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: an array of functions";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: An array of shapes.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: an array of dictionaries";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tfg::RegionAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: case region preserved attributes";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::tf_type::FuncAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DictionaryAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: dictionary of named attribute values";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::tfg::RegionAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Region attributes, argument and result attributes.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::tf_type::VersionAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: An Attribute describing the version for a TensorFlow Graph";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops17(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_ops18(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_ops0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_ops1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::CaseOp definitions
//===----------------------------------------------------------------------===//

CaseOpAdaptor::CaseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.Case", odsAttrs.getContext());
}

CaseOpAdaptor::CaseOpAdaptor(CaseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CaseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CaseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange CaseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CaseOpAdaptor::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CaseOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr CaseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CaseOpAdaptor::getBranchesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CaseOp::getBranchesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CaseOpAdaptor::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::mlir::ArrayAttr CaseOpAdaptor::getTinAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CaseOp::getTinAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CaseOpAdaptor::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CaseOpAdaptor::getToutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CaseOp::getToutAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CaseOpAdaptor::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CaseOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, CaseOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CaseOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult CaseOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branches;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.Case' op ""requires attribute 'branches'");
    if (namedAttrIt->getName() == CaseOp::getBranchesAttrName(*odsOpName)) {
      tblgen_branches = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CaseOp::getTinAttrName(*odsOpName)) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CaseOp::getToutAttrName(*odsOpName)) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CaseOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_branches && !(((tblgen_branches.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_branches.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::FuncAttr>())); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'branches' failed to satisfy constraint: an array of functions");

  if (tblgen_Tin && !(((tblgen_Tin.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tin.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((tblgen_Tout.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tout.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.Case' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void CaseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> CaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range CaseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CaseOp::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CaseOp::getArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange CaseOp::getBranchIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CaseOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CaseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range CaseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CaseOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> CaseOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::ArrayAttr CaseOp::getBranchesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CaseOp::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::mlir::ArrayAttr CaseOp::getTinAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getTinAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CaseOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CaseOp::getToutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getToutAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CaseOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CaseOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CaseOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void CaseOp::setBranchesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBranchesAttrName(), attr);
}

void CaseOp::setTinAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTinAttrName(), attr);
}

void CaseOp::setToutAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getToutAttrName(), attr);
}

void CaseOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute CaseOp::removeTinAttr() {
  return (*this)->removeAttr(getTinAttrName());
}

::mlir::Attribute CaseOp::removeToutAttr() {
  return (*this)->removeAttr(getToutAttrName());
}

::mlir::Attribute CaseOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void CaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches) {
      build(odsBuilder, odsState, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void CaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.addAttribute(getBranchesAttrName(odsState.name), branches);
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void CaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.addAttribute(getBranchesAttrName(odsState.name), branches);
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CaseOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branches;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'branches'");
    if (namedAttrIt->getName() == getBranchesAttrName()) {
      tblgen_branches = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTinAttrName()) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getToutAttrName()) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops0(*this, tblgen_branches, "branches")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult CaseOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyCaseLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::CaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::CaseRegionOp definitions
//===----------------------------------------------------------------------===//

CaseRegionOpAdaptor::CaseRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.CaseRegion", odsAttrs.getContext());
}

CaseRegionOpAdaptor::CaseRegionOpAdaptor(CaseRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CaseRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CaseRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange CaseRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CaseRegionOpAdaptor::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CaseRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr CaseRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CaseRegionOpAdaptor::getBranchAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CaseRegionOp::getBranchAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CaseRegionOpAdaptor::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CaseRegionOpAdaptor::getRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CaseRegionOp::getRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > CaseRegionOpAdaptor::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::RegionRange CaseRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::RegionRange CaseRegionOpAdaptor::getBranches() {
  return odsRegions.drop_front(0);
}

::mlir::LogicalResult CaseRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_attrs;
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CaseRegionOp::getBranchAttrsAttrName(*odsOpName)) {
      tblgen_branch_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CaseRegionOp::getRegionAttrsAttrName(*odsOpName)) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_branch_attrs && !(((tblgen_branch_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_branch_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'tfg.CaseRegion' op ""attribute 'branch_attrs' failed to satisfy constraint: an array of dictionaries");

  if (tblgen_region_attrs && !(((tblgen_region_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_region_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tfg::RegionAttr>())); }))))
    return emitError(loc, "'tfg.CaseRegion' op ""attribute 'region_attrs' failed to satisfy constraint: case region preserved attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CaseRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range CaseRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> CaseRegionOp::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CaseRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange CaseRegionOp::getBranchIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CaseRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CaseRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range CaseRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range CaseRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> CaseRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::MutableArrayRef<::mlir::Region> CaseRegionOp::getBranches() {
  return (*this)->getRegions().drop_front(0);
}

::mlir::ArrayAttr CaseRegionOp::getBranchAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CaseRegionOp::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr CaseRegionOp::getRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getRegionAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > CaseRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void CaseRegionOp::setBranchAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBranchAttrsAttrName(), attr);
}

void CaseRegionOp::setRegionAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getRegionAttrsAttrName(), attr);
}

::mlir::Attribute CaseRegionOp::removeBranch_attrsAttr() {
  return (*this)->removeAttr(getBranchAttrsAttrName());
}

::mlir::Attribute CaseRegionOp::removeRegion_attrsAttr() {
  return (*this)->removeAttr(getRegionAttrsAttrName());
}

void CaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.addAttribute(getBranchAttrsAttrName(odsState.name), branch_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void CaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.addAttribute(getBranchAttrsAttrName(odsState.name), branch_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CaseRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CaseRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_attrs;
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBranchAttrsAttrName()) {
      tblgen_branch_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getRegionAttrsAttrName()) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_branch_attrs, "branch_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops4(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getBranches())
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "branches", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult CaseRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CaseRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand branch_indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> branch_indexOperands(branch_indexRawOperands);  ::llvm::SMLoc branch_indexOperandsLoc;
  (void)branch_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> branchesRegions;
  ::llvm::ArrayRef<::mlir::Type> branch_indexTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  branch_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(branch_indexRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      branchesRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        branchesRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : branchesRegions)
    if (region->empty()) region->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType branch_index__outs_functionType;
  if (parser.parseType(branch_index__outs_functionType))
    return ::mlir::failure();
  branch_indexTypes = branch_index__outs_functionType.getInputs();
  outsTypes = branch_index__outs_functionType.getResults();
  result.addRegions(branchesRegions);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(branch_indexOperands, branch_indexTypes, branch_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CaseRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBranchIndex();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
    llvm::interleaveComma(getBranches(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getBranchIndex().getType()), getOuts().getTypes());
}

StringRef CaseRegionOp::getDefaultDialect() { return "tfg"; }
void CaseRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp CaseRegionOp::branch_yield(unsigned idx) {
  return cast<YieldOp>(getBranchBlock(idx).getTerminator());
}

void CaseRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpSuccessorRegions(*this, index, operands, regions);
}
void CaseRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
}

LogicalResult CaseRegionOp::verify() { return VerifyCaseLikeRegionOp(*this); }

BlockArgument CaseRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument CaseRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument CaseRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument CaseRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::CaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ConditionOp definitions
//===----------------------------------------------------------------------===//

ConditionOpAdaptor::ConditionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.condition", odsAttrs.getContext());
}

ConditionOpAdaptor::ConditionOpAdaptor(ConditionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConditionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConditionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConditionOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ConditionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConditionOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange ConditionOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::ValueRange ConditionOpAdaptor::getCtls() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr ConditionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ConditionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.condition' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ConditionOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'tfg.condition' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> ConditionOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ConditionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ConditionOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range ConditionOp::getArgs() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range ConditionOp::getCtls() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange ConditionOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ConditionOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ConditionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ConditionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConditionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ConditionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}));
}

void ConditionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConditionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConditionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConditionOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ConditionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(condRawOperands);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::mlir::Type condRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> condTypes(condRawTypes);
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    condRawTypes[0] = type;
  }
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(argsOperands.size()), static_cast<int32_t>(ctlsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConditionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getCond().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getArgs().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  if (!getArgs().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
  }
}

void ConditionOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ConditionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ForOp definitions
//===----------------------------------------------------------------------===//

ForOpAdaptor::ForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.For", odsAttrs.getContext());
}

ForOpAdaptor::ForOpAdaptor(ForOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ForOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ForOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ForOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ForOpAdaptor::getStart() {
  return *getODSOperands(0).begin();
}

::mlir::Value ForOpAdaptor::getLimit() {
  return *getODSOperands(1).begin();
}

::mlir::Value ForOpAdaptor::getDelta() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange ForOpAdaptor::getArgs() {
  return getODSOperands(3);
}

::mlir::DictionaryAttr ForOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr ForOpAdaptor::getBodyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ForOp::getBodyAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr ForOpAdaptor::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::ArrayAttr ForOpAdaptor::getTAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ForOp::getTAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > ForOpAdaptor::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult ForOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.For' op ""requires attribute 'body'");
    if (namedAttrIt->getName() == ForOp::getBodyAttrName(*odsOpName)) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ForOp::getTAttrName(*odsOpName)) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_body && !((tblgen_body.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.For' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_T && !(((tblgen_T.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_T.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.For' op ""attribute 'T' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

void ForOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> ForOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 3) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ForOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ForOp::getStart() {
  return *getODSOperands(0).begin();
}

::mlir::Value ForOp::getLimit() {
  return *getODSOperands(1).begin();
}

::mlir::Value ForOp::getDelta() {
  return *getODSOperands(2).begin();
}

::mlir::Operation::operand_range ForOp::getArgs() {
  return getODSOperands(3);
}

::mlir::MutableOperandRange ForOp::getStartMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ForOp::getLimitMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ForOp::getDeltaMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ForOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ForOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ForOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ForOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> ForOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr ForOp::getBodyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBodyAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr ForOp::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::ArrayAttr ForOp::getTAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getTAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ForOp::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void ForOp::setBodyAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getBodyAttrName(), attr);
}

void ForOp::setTAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTAttrName(), attr);
}

::mlir::Attribute ForOp::removeTAttr() {
  return (*this)->removeAttr(getTAttrName());
}

void ForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value start, Value limit, Value delta, ValueRange args, FuncAttr body) {
      build(odsBuilder, odsState, results, start, limit, delta, args, body,
            /*T=*/{});
    
}

void ForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr body, /*optional*/::mlir::ArrayAttr T) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(args);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void ForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr body, /*optional*/::mlir::ArrayAttr T) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(args);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ForOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'body'");
    if (namedAttrIt->getName() == getBodyAttrName()) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTAttrName()) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ForOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ForRegionOp definitions
//===----------------------------------------------------------------------===//

ForRegionOpAdaptor::ForRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.ForRegion", odsAttrs.getContext());
}

ForRegionOpAdaptor::ForRegionOpAdaptor(ForRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ForRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ForRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ForRegionOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange ForRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ForRegionOpAdaptor::getStart() {
  return *getODSOperands(0).begin();
}

::mlir::Value ForRegionOpAdaptor::getLimit() {
  return *getODSOperands(1).begin();
}

::mlir::Value ForRegionOpAdaptor::getDelta() {
  return *getODSOperands(2).begin();
}

::mlir::ValueRange ForRegionOpAdaptor::getInit() {
  return getODSOperands(3);
}

::mlir::ValueRange ForRegionOpAdaptor::getCtls() {
  return getODSOperands(4);
}

::mlir::DictionaryAttr ForRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DictionaryAttr ForRegionOpAdaptor::getBodyAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ForRegionOp::getBodyAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > ForRegionOpAdaptor::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr ForRegionOpAdaptor::getRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ForRegionOp::getRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> ForRegionOpAdaptor::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange ForRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ForRegionOpAdaptor::getBodyRegion() {
  return *odsRegions[0];
}

::mlir::LogicalResult ForRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.ForRegion' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ForRegionOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ForRegionOp::getBodyAttrsAttrName(*odsOpName)) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ForRegionOp::getRegionAttrsAttrName(*odsOpName)) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitError(loc, "'tfg.ForRegion' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_body_attrs && !((tblgen_body_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.ForRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_region_attrs && !((tblgen_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.ForRegion' op ""attribute 'region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ForRegionOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range ForRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ForRegionOp::getStart() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::TensorType> ForRegionOp::getLimit() {
  return *getODSOperands(1).begin();
}

::mlir::TypedValue<::mlir::TensorType> ForRegionOp::getDelta() {
  return *getODSOperands(2).begin();
}

::mlir::Operation::operand_range ForRegionOp::getInit() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ForRegionOp::getCtls() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange ForRegionOp::getStartMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ForRegionOp::getLimitMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ForRegionOp::getDeltaMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ForRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ForRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ForRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range ForRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ForRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> ForRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &ForRegionOp::getBodyRegion() {
  return (*this)->getRegion(0);
}

::mlir::DictionaryAttr ForRegionOp::getBodyAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBodyAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > ForRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr ForRegionOp::getRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> ForRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void ForRegionOp::setBodyAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getBodyAttrsAttrName(), attr);
}

void ForRegionOp::setRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getRegionAttrsAttrName(), attr);
}

::mlir::Attribute ForRegionOp::removeBody_attrsAttr() {
  return (*this)->removeAttr(getBodyAttrsAttrName());
}

::mlir::Attribute ForRegionOp::removeRegion_attrsAttr() {
  return (*this)->removeAttr(getRegionAttrsAttrName());
}

void ForRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange init, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr region_attrs) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, 1, static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void ForRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange init, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr region_attrs) {
  odsState.addOperands(start);
  odsState.addOperands(limit);
  odsState.addOperands(delta);
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, 1, static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ForRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBodyAttrsAttrName()) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getRegionAttrsAttrName()) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ForRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand startRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> startOperands(startRawOperands);  ::llvm::SMLoc startOperandsLoc;
  (void)startOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand limitRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> limitOperands(limitRawOperands);  ::llvm::SMLoc limitOperandsLoc;
  (void)limitOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand deltaRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> deltaOperands(deltaRawOperands);  ::llvm::SMLoc deltaOperandsLoc;
  (void)deltaOperandsLoc;
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::mlir::Type startRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> startTypes(startRawTypes);
  ::mlir::Type limitRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> limitTypes(limitRawTypes);
  ::mlir::Type deltaRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> deltaTypes(deltaRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> initTypes;
  ::llvm::SmallVector<::mlir::Type, 1> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("from"))
    return ::mlir::failure();

  startOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(startRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  limitOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(limitRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("by"))
    return ::mlir::failure();

  deltaOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(deltaRawOperands[0]))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    startRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    limitRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    deltaRawTypes[0] = type;
  }
  if (::mlir::succeeded(parser.parseOptionalComma())) {

  if (parser.parseTypeList(initTypes))
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalArrow())) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseTypeList(outsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  result.addRegion(std::move(body_regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, 1, 1, static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(startOperands, startTypes, startOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(limitOperands, limitTypes, limitOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deltaOperands, deltaTypes, deltaOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ForRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "from";
  _odsPrinter << ' ';
  _odsPrinter << getStart();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getLimit();
  _odsPrinter << ' ' << "by";
  _odsPrinter << ' ';
  _odsPrinter << getDelta();
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getStart().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getLimit().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getDelta().getType();
    if (auto validType = type.dyn_cast<::mlir::TensorType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  if (!getInit().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    _odsPrinter << getInit().getTypes();
  }
  _odsPrinter << ")";
  if (!getOuts().empty()) {
    _odsPrinter << ' ' << "->";
    _odsPrinter << ' ' << "(";
    _odsPrinter << getOuts().getTypes();
    _odsPrinter << ")";
  }
}

StringRef ForRegionOp::getDefaultDialect() { return "tfg"; }
void ForRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp ForRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ForRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GetResultOp definitions
//===----------------------------------------------------------------------===//

GetResultOpAdaptor::GetResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.get_result", odsAttrs.getContext());
}

GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GetResultOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetResultOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetResultOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetResultOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetResultOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GetResultOpAdaptor::getNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GetResultOp::getNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GetResultOpAdaptor::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr GetResultOpAdaptor::getNumberAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GetResultOp::getNumberAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetResultOpAdaptor::getNumber() {
  auto attr = getNumberAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.get_result' op ""requires attribute 'name'");
    if (namedAttrIt->getName() == GetResultOp::getNameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_number;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.get_result' op ""requires attribute 'number'");
    if (namedAttrIt->getName() == GetResultOp::getNumberAttrName(*odsOpName)) {
      tblgen_number = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'tfg.get_result' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_number && !(((tblgen_number.isa<::mlir::IntegerAttr>())) && ((tblgen_number.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'tfg.get_result' op ""attribute 'number' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetResultOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::tf_type::OpaqueTensorType> GetResultOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GetResultOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::tf_type::OpaqueTensorType> GetResultOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::StringAttr GetResultOp::getNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GetResultOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr GetResultOp::getNumberAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getNumberAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t GetResultOp::getNumber() {
  auto attr = getNumberAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void GetResultOp::setNumberAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumberAttrName(), attr);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number) {
  odsState.addOperands(value);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getNumberAttrName(odsState.name), number);
  odsState.addTypes(result);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number) {
  odsState.addOperands(value);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getNumberAttrName(odsState.name), number);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(GetResultOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number) {
  odsState.addOperands(value);
  odsState.addAttribute(getNameAttrName(odsState.name), name);
  odsState.addAttribute(getNumberAttrName(odsState.name), number);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::llvm::StringRef name, uint32_t number) {
  odsState.addOperands(value);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getNumberAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), number));
  odsState.addTypes(result);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::llvm::StringRef name, uint32_t number) {
  odsState.addOperands(value);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getNumberAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), number));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(GetResultOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::llvm::StringRef name, uint32_t number) {
  odsState.addOperands(value);
  odsState.addAttribute(getNameAttrName(odsState.name), odsBuilder.getStringAttr(name));
  odsState.addAttribute(getNumberAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), number));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(GetResultOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult GetResultOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'name'");
    if (namedAttrIt->getName() == getNameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_number;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'number'");
    if (namedAttrIt->getName() == getNumberAttrName()) {
      tblgen_number = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops8(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops9(*this, tblgen_number, "number")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetResultOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult GetResultOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = odsBuilder.getType<::mlir::tf_type::OpaqueTensorType>();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult GetResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::mlir::IntegerAttr numberAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(numberAttr, parser.getBuilder().getIntegerType(32), "number",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::OpaqueTensorType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getValue();
  _odsPrinter << ")";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNumberAttr());
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"name", "number"});
}

void GetResultOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::GetResultOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GraphFuncOp definitions
//===----------------------------------------------------------------------===//

GraphFuncOpAdaptor::GraphFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.func", odsAttrs.getContext());
}

GraphFuncOpAdaptor::GraphFuncOpAdaptor(GraphFuncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GraphFuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GraphFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GraphFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GraphFuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr GraphFuncOpAdaptor::getGenericAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GraphFuncOp::getGenericAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GraphFuncOpAdaptor::getGeneric() {
  auto attr = getGenericAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::StringAttr GraphFuncOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GraphFuncOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GraphFuncOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr GraphFuncOpAdaptor::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GraphFuncOp::getFunctionTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::FunctionType GraphFuncOpAdaptor::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::ArrayAttr GraphFuncOpAdaptor::getArgAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GraphFuncOp::getArgAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > GraphFuncOpAdaptor::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr GraphFuncOpAdaptor::getResAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GraphFuncOp::getResAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > GraphFuncOpAdaptor::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::StringAttr GraphFuncOpAdaptor::getDescriptionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GraphFuncOp::getDescriptionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GraphFuncOpAdaptor::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr GraphFuncOpAdaptor::getIsStatefulAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GraphFuncOp::getIsStatefulAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool GraphFuncOpAdaptor::getIsStateful() {
  auto attr = getIsStatefulAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::FlatSymbolRefAttr GraphFuncOpAdaptor::getGradientAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GraphFuncOp::getGradientAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
  return attr;
}

::llvm::Optional< ::llvm::StringRef > GraphFuncOpAdaptor::getGradient() {
  auto attr = getGradientAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::DenseIntElementsAttr GraphFuncOpAdaptor::getResourceArgUniqueIdsKeysAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GraphFuncOp::getResourceArgUniqueIdsKeysAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > GraphFuncOpAdaptor::getResourceArgUniqueIdsKeys() {
  auto attr = getResourceArgUniqueIdsKeysAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr GraphFuncOpAdaptor::getResourceArgUniqueIdsValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GraphFuncOp::getResourceArgUniqueIdsValuesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > GraphFuncOpAdaptor::getResourceArgUniqueIdsValues() {
  auto attr = getResourceArgUniqueIdsValuesAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr GraphFuncOpAdaptor::getControlOutputAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GraphFuncOp::getControlOutputAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > GraphFuncOpAdaptor::getControlOutput() {
  auto attr = getControlOutputAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::RegionRange GraphFuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GraphFuncOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult GraphFuncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_arg_attrs;
  ::mlir::Attribute tblgen_control_output;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.func' op ""requires attribute 'function_type'");
    if (namedAttrIt->getName() == GraphFuncOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getArgAttrsAttrName(*odsOpName)) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getControlOutputAttrName(*odsOpName)) {
      tblgen_control_output = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getDescriptionAttrName(*odsOpName)) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_generic;
  ::mlir::Attribute tblgen_gradient;
  ::mlir::Attribute tblgen_is_stateful;
  ::mlir::Attribute tblgen_res_attrs;
  ::mlir::Attribute tblgen_resource_arg_unique_ids_keys;
  ::mlir::Attribute tblgen_resource_arg_unique_ids_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.func' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == GraphFuncOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getGenericAttrName(*odsOpName)) {
      tblgen_generic = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getGradientAttrName(*odsOpName)) {
      tblgen_gradient = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getIsStatefulAttrName(*odsOpName)) {
      tblgen_is_stateful = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getResAttrsAttrName(*odsOpName)) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getResourceArgUniqueIdsKeysAttrName(*odsOpName)) {
      tblgen_resource_arg_unique_ids_keys = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GraphFuncOp::getResourceArgUniqueIdsValuesAttrName(*odsOpName)) {
      tblgen_resource_arg_unique_ids_values = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_generic && !((tblgen_generic.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'generic' failed to satisfy constraint: unit attribute");

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>()))))
    return emitError(loc, "'tfg.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_arg_attrs && !((tblgen_arg_attrs.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'arg_attrs' failed to satisfy constraint: array attribute");

  if (tblgen_res_attrs && !((tblgen_res_attrs.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'res_attrs' failed to satisfy constraint: array attribute");

  if (tblgen_description && !((tblgen_description.isa<::mlir::StringAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'description' failed to satisfy constraint: string attribute");

  if (tblgen_is_stateful && !((tblgen_is_stateful.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'is_stateful' failed to satisfy constraint: unit attribute");

  if (tblgen_gradient && !((tblgen_gradient.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'tfg.func' op ""attribute 'gradient' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_resource_arg_unique_ids_keys && !(((tblgen_resource_arg_unique_ids_keys.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_resource_arg_unique_ids_keys.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'tfg.func' op ""attribute 'resource_arg_unique_ids_keys' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_resource_arg_unique_ids_values && !(((tblgen_resource_arg_unique_ids_values.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_resource_arg_unique_ids_values.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'tfg.func' op ""attribute 'resource_arg_unique_ids_values' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_control_output && !(((tblgen_control_output.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_control_output.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'tfg.func' op ""attribute 'control_output' failed to satisfy constraint: string array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GraphFuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GraphFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GraphFuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GraphFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &GraphFuncOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr GraphFuncOp::getGenericAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getGenericAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GraphFuncOp::getGeneric() {
  auto attr = getGenericAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::StringAttr GraphFuncOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GraphFuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr GraphFuncOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::FunctionType GraphFuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::ArrayAttr GraphFuncOp::getArgAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getArgAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > GraphFuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr GraphFuncOp::getResAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getResAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > GraphFuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::StringAttr GraphFuncOp::getDescriptionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDescriptionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GraphFuncOp::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr GraphFuncOp::getIsStatefulAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIsStatefulAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GraphFuncOp::getIsStateful() {
  auto attr = getIsStatefulAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

::mlir::FlatSymbolRefAttr GraphFuncOp::getGradientAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getGradientAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > GraphFuncOp::getGradient() {
  auto attr = getGradientAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::DenseIntElementsAttr GraphFuncOp::getResourceArgUniqueIdsKeysAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getResourceArgUniqueIdsKeysAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > GraphFuncOp::getResourceArgUniqueIdsKeys() {
  auto attr = getResourceArgUniqueIdsKeysAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::DenseIntElementsAttr GraphFuncOp::getResourceArgUniqueIdsValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getResourceArgUniqueIdsValuesAttrName()).dyn_cast_or_null<::mlir::DenseIntElementsAttr>();
}

::llvm::Optional< ::mlir::DenseIntElementsAttr > GraphFuncOp::getResourceArgUniqueIdsValues() {
  auto attr = getResourceArgUniqueIdsValuesAttr();
  return attr ? ::llvm::Optional< ::mlir::DenseIntElementsAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr GraphFuncOp::getControlOutputAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getControlOutputAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > GraphFuncOp::getControlOutput() {
  auto attr = getControlOutputAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void GraphFuncOp::setGenericAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getGenericAttrName(), attr);
}

void GraphFuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void GraphFuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void GraphFuncOp::setArgAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgAttrsAttrName(), attr);
}

void GraphFuncOp::setResAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getResAttrsAttrName(), attr);
}

void GraphFuncOp::setDescriptionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDescriptionAttrName(), attr);
}

void GraphFuncOp::setIsStatefulAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIsStatefulAttrName(), attr);
}

void GraphFuncOp::setGradientAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getGradientAttrName(), attr);
}

void GraphFuncOp::setResourceArgUniqueIdsKeysAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getResourceArgUniqueIdsKeysAttrName(), attr);
}

void GraphFuncOp::setResourceArgUniqueIdsValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getResourceArgUniqueIdsValuesAttrName(), attr);
}

void GraphFuncOp::setControlOutputAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getControlOutputAttrName(), attr);
}

::mlir::Attribute GraphFuncOp::removeGenericAttr() {
  return (*this)->removeAttr(getGenericAttrName());
}

::mlir::Attribute GraphFuncOp::removeArg_attrsAttr() {
  return (*this)->removeAttr(getArgAttrsAttrName());
}

::mlir::Attribute GraphFuncOp::removeRes_attrsAttr() {
  return (*this)->removeAttr(getResAttrsAttrName());
}

::mlir::Attribute GraphFuncOp::removeDescriptionAttr() {
  return (*this)->removeAttr(getDescriptionAttrName());
}

::mlir::Attribute GraphFuncOp::removeIs_statefulAttr() {
  return (*this)->removeAttr(getIsStatefulAttrName());
}

::mlir::Attribute GraphFuncOp::removeGradientAttr() {
  return (*this)->removeAttr(getGradientAttrName());
}

::mlir::Attribute GraphFuncOp::removeResource_arg_unique_ids_keysAttr() {
  return (*this)->removeAttr(getResourceArgUniqueIdsKeysAttrName());
}

::mlir::Attribute GraphFuncOp::removeResource_arg_unique_ids_valuesAttr() {
  return (*this)->removeAttr(getResourceArgUniqueIdsValuesAttrName());
}

::mlir::Attribute GraphFuncOp::removeControl_outputAttr() {
  return (*this)->removeAttr(getControlOutputAttrName());
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 (void)odsState.addRegion(); 
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, const Twine &sym_name, FunctionType type, bool generic) {
      build(odsBuilder, odsState);
      odsState.addAttribute(GraphFuncOp::getSymNameAttrName(odsState.name),
                           odsBuilder.getStringAttr(sym_name));
      odsState.addAttribute(GraphFuncOp::getFunctionTypeAttrName(odsState.name),
                           TypeAttr::get(type));
      if (generic) {
        odsState.addAttribute(GraphFuncOp::getGenericAttrName(odsState.name),
                             odsBuilder.getUnitAttr());
      }
    
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr generic, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/::mlir::UnitAttr is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.addAttribute(getGenericAttrName(odsState.name), generic);
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), function_type);
  if (arg_attrs) {
    odsState.addAttribute(getArgAttrsAttrName(odsState.name), arg_attrs);
  }
  if (res_attrs) {
    odsState.addAttribute(getResAttrsAttrName(odsState.name), res_attrs);
  }
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  if (is_stateful) {
    odsState.addAttribute(getIsStatefulAttrName(odsState.name), is_stateful);
  }
  if (gradient) {
    odsState.addAttribute(getGradientAttrName(odsState.name), gradient);
  }
  if (resource_arg_unique_ids_keys) {
    odsState.addAttribute(getResourceArgUniqueIdsKeysAttrName(odsState.name), resource_arg_unique_ids_keys);
  }
  if (resource_arg_unique_ids_values) {
    odsState.addAttribute(getResourceArgUniqueIdsValuesAttrName(odsState.name), resource_arg_unique_ids_values);
  }
  if (control_output) {
    odsState.addAttribute(getControlOutputAttrName(odsState.name), control_output);
  }
  (void)odsState.addRegion();
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr generic, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/::mlir::UnitAttr is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.addAttribute(getGenericAttrName(odsState.name), generic);
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), function_type);
  if (arg_attrs) {
    odsState.addAttribute(getArgAttrsAttrName(odsState.name), arg_attrs);
  }
  if (res_attrs) {
    odsState.addAttribute(getResAttrsAttrName(odsState.name), res_attrs);
  }
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  if (is_stateful) {
    odsState.addAttribute(getIsStatefulAttrName(odsState.name), is_stateful);
  }
  if (gradient) {
    odsState.addAttribute(getGradientAttrName(odsState.name), gradient);
  }
  if (resource_arg_unique_ids_keys) {
    odsState.addAttribute(getResourceArgUniqueIdsKeysAttrName(odsState.name), resource_arg_unique_ids_keys);
  }
  if (resource_arg_unique_ids_values) {
    odsState.addAttribute(getResourceArgUniqueIdsValuesAttrName(odsState.name), resource_arg_unique_ids_values);
  }
  if (control_output) {
    odsState.addAttribute(getControlOutputAttrName(odsState.name), control_output);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool generic, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/bool is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.addAttribute(getGenericAttrName(odsState.name), ((generic) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), ::mlir::TypeAttr::get(function_type));
  if (arg_attrs) {
    odsState.addAttribute(getArgAttrsAttrName(odsState.name), arg_attrs);
  }
  if (res_attrs) {
    odsState.addAttribute(getResAttrsAttrName(odsState.name), res_attrs);
  }
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  if (is_stateful) {
    odsState.addAttribute(getIsStatefulAttrName(odsState.name), ((is_stateful) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (gradient) {
    odsState.addAttribute(getGradientAttrName(odsState.name), gradient);
  }
  if (resource_arg_unique_ids_keys) {
    odsState.addAttribute(getResourceArgUniqueIdsKeysAttrName(odsState.name), resource_arg_unique_ids_keys);
  }
  if (resource_arg_unique_ids_values) {
    odsState.addAttribute(getResourceArgUniqueIdsValuesAttrName(odsState.name), resource_arg_unique_ids_values);
  }
  if (control_output) {
    odsState.addAttribute(getControlOutputAttrName(odsState.name), control_output);
  }
  (void)odsState.addRegion();
}

void GraphFuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool generic, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/bool is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output) {
  if (generic) {
    odsState.addAttribute(getGenericAttrName(odsState.name), ((generic) ? odsBuilder.getUnitAttr() : nullptr));
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), ::mlir::TypeAttr::get(function_type));
  if (arg_attrs) {
    odsState.addAttribute(getArgAttrsAttrName(odsState.name), arg_attrs);
  }
  if (res_attrs) {
    odsState.addAttribute(getResAttrsAttrName(odsState.name), res_attrs);
  }
  if (description) {
    odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  if (is_stateful) {
    odsState.addAttribute(getIsStatefulAttrName(odsState.name), ((is_stateful) ? odsBuilder.getUnitAttr() : nullptr));
  }
  if (gradient) {
    odsState.addAttribute(getGradientAttrName(odsState.name), gradient);
  }
  if (resource_arg_unique_ids_keys) {
    odsState.addAttribute(getResourceArgUniqueIdsKeysAttrName(odsState.name), resource_arg_unique_ids_keys);
  }
  if (resource_arg_unique_ids_values) {
    odsState.addAttribute(getResourceArgUniqueIdsValuesAttrName(odsState.name), resource_arg_unique_ids_values);
  }
  if (control_output) {
    odsState.addAttribute(getControlOutputAttrName(odsState.name), control_output);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GraphFuncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GraphFuncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_arg_attrs;
  ::mlir::Attribute tblgen_control_output;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getArgAttrsAttrName()) {
      tblgen_arg_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getControlOutputAttrName()) {
      tblgen_control_output = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDescriptionAttrName()) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_generic;
  ::mlir::Attribute tblgen_gradient;
  ::mlir::Attribute tblgen_is_stateful;
  ::mlir::Attribute tblgen_res_attrs;
  ::mlir::Attribute tblgen_resource_arg_unique_ids_keys;
  ::mlir::Attribute tblgen_resource_arg_unique_ids_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getGenericAttrName()) {
      tblgen_generic = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getGradientAttrName()) {
      tblgen_gradient = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIsStatefulAttrName()) {
      tblgen_is_stateful = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getResAttrsAttrName()) {
      tblgen_res_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getResourceArgUniqueIdsKeysAttrName()) {
      tblgen_resource_arg_unique_ids_keys = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getResourceArgUniqueIdsValuesAttrName()) {
      tblgen_resource_arg_unique_ids_values = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops10(*this, tblgen_generic, "generic")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops8(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops11(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops12(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops12(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops8(*this, tblgen_description, "description")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops10(*this, tblgen_is_stateful, "is_stateful")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops13(*this, tblgen_gradient, "gradient")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops14(*this, tblgen_resource_arg_unique_ids_keys, "resource_arg_unique_ids_keys")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops14(*this, tblgen_resource_arg_unique_ids_values, "resource_arg_unique_ids_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops15(*this, tblgen_control_output, "control_output")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GraphFuncOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void GraphFuncOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.add(canonicalize);
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::GraphFuncOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GraphOp definitions
//===----------------------------------------------------------------------===//

GraphOpAdaptor::GraphOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.graph", odsAttrs.getContext());
}

GraphOpAdaptor::GraphOpAdaptor(GraphOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GraphOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GraphOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GraphOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GraphOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::VersionAttr GraphOpAdaptor::getVersionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GraphOp::getVersionAttrName(*odsOpName)).cast<::mlir::tf_type::VersionAttr>();
  return attr;
}

::mlir::tf_type::VersionAttr GraphOpAdaptor::getVersion() {
  auto attr = getVersionAttr();
  return attr.cast<::mlir::tf_type::VersionAttr>();
}

::mlir::RegionRange GraphOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GraphOpAdaptor::getNodes() {
  return *odsRegions[0];
}

::mlir::LogicalResult GraphOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_version;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.graph' op ""requires attribute 'version'");
    if (namedAttrIt->getName() == GraphOp::getVersionAttrName(*odsOpName)) {
      tblgen_version = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_version && !((tblgen_version.isa<::mlir::tf_type::VersionAttr>())))
    return emitError(loc, "'tfg.graph' op ""attribute 'version' failed to satisfy constraint: An Attribute describing the version for a TensorFlow Graph");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GraphOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GraphOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GraphOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GraphOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &GraphOp::getNodes() {
  return (*this)->getRegion(0);
}

::mlir::tf_type::VersionAttr GraphOp::getVersionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVersionAttrName()).cast<::mlir::tf_type::VersionAttr>();
}

::mlir::tf_type::VersionAttr GraphOp::getVersion() {
  auto attr = getVersionAttr();
  return attr.cast<::mlir::tf_type::VersionAttr>();
}

void GraphOp::setVersionAttr(::mlir::tf_type::VersionAttr attr) {
  (*this)->setAttr(getVersionAttrName(), attr);
}

void GraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::tf_type::VersionAttr version) {
  odsState.addAttribute(getVersionAttrName(odsState.name), version);
  (void)odsState.addRegion();
}

void GraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::tf_type::VersionAttr version) {
  odsState.addAttribute(getVersionAttrName(odsState.name), version);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GraphOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GraphOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_version;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'version'");
    if (namedAttrIt->getName() == getVersionAttrName()) {
      tblgen_version = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops16(*this, tblgen_version, "version")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "nodes", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GraphOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GraphOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::tf_type::VersionAttr versionAttr;
  std::unique_ptr<::mlir::Region> nodesRegion = std::make_unique<::mlir::Region>();

  if (parser.parseAttribute(versionAttr, ::mlir::Type{}, "version", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*nodesRegion))
    return ::mlir::failure();

  if (nodesRegion->empty()) nodesRegion->emplaceBlock();
  result.addRegion(std::move(nodesRegion));
  return ::mlir::success();
}

void GraphOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getVersionAttr());
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"version"});
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getNodes());
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::GraphOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::IfOp definitions
//===----------------------------------------------------------------------===//

IfOpAdaptor::IfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.If", odsAttrs.getContext());
}

IfOpAdaptor::IfOpAdaptor(IfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange IfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IfOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange IfOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr IfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr IfOpAdaptor::getThenBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, IfOp::getThenBranchAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr IfOpAdaptor::getThenBranch() {
  auto attr = getThenBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr IfOpAdaptor::getElseBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, IfOp::getElseBranchAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr IfOpAdaptor::getElseBranch() {
  auto attr = getElseBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::TypeAttr IfOpAdaptor::getTcondAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, IfOp::getTcondAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::llvm::Optional<::mlir::Type> IfOpAdaptor::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::ArrayAttr IfOpAdaptor::getTinAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, IfOp::getTinAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > IfOpAdaptor::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr IfOpAdaptor::getToutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, IfOp::getToutAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > IfOpAdaptor::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr IfOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, IfOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > IfOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult IfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_branch;
  ::mlir::Attribute tblgen_Tcond;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.If' op ""requires attribute 'else_branch'");
    if (namedAttrIt->getName() == IfOp::getElseBranchAttrName(*odsOpName)) {
      tblgen_else_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == IfOp::getTcondAttrName(*odsOpName)) {
      tblgen_Tcond = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == IfOp::getTinAttrName(*odsOpName)) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == IfOp::getToutAttrName(*odsOpName)) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_then_branch;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.If' op ""requires attribute 'then_branch'");
    if (namedAttrIt->getName() == IfOp::getThenBranchAttrName(*odsOpName)) {
      tblgen_then_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == IfOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_then_branch && !((tblgen_then_branch.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.If' op ""attribute 'then_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_else_branch && !((tblgen_else_branch.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.If' op ""attribute 'else_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_Tcond && !(((tblgen_Tcond.isa<::mlir::TypeAttr>())) && ((tblgen_Tcond.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'tfg.If' op ""attribute 'Tcond' failed to satisfy constraint: any type attribute");

  if (tblgen_Tin && !(((tblgen_Tin.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tin.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.If' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((tblgen_Tout.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tout.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.If' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.If' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void IfOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> IfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range IfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IfOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range IfOp::getArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange IfOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange IfOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range IfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IfOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> IfOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr IfOp::getThenBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getThenBranchAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr IfOp::getThenBranch() {
  auto attr = getThenBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr IfOp::getElseBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getElseBranchAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr IfOp::getElseBranch() {
  auto attr = getElseBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::TypeAttr IfOp::getTcondAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getTcondAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> IfOp::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::ArrayAttr IfOp::getTinAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getTinAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > IfOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr IfOp::getToutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getToutAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > IfOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr IfOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > IfOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void IfOp::setThenBranchAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getThenBranchAttrName(), attr);
}

void IfOp::setElseBranchAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getElseBranchAttrName(), attr);
}

void IfOp::setTcondAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTcondAttrName(), attr);
}

void IfOp::setTinAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTinAttrName(), attr);
}

void IfOp::setToutAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getToutAttrName(), attr);
}

void IfOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute IfOp::removeTcondAttr() {
  return (*this)->removeAttr(getTcondAttrName());
}

::mlir::Attribute IfOp::removeTinAttr() {
  return (*this)->removeAttr(getTinAttrName());
}

::mlir::Attribute IfOp::removeToutAttr() {
  return (*this)->removeAttr(getToutAttrName());
}

::mlir::Attribute IfOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch) {
      build(odsBuilder, odsState, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addAttribute(getThenBranchAttrName(odsState.name), then_branch);
  odsState.addAttribute(getElseBranchAttrName(odsState.name), else_branch);
  if (Tcond) {
    odsState.addAttribute(getTcondAttrName(odsState.name), Tcond);
  }
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void IfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addAttribute(getThenBranchAttrName(odsState.name), then_branch);
  odsState.addAttribute(getElseBranchAttrName(odsState.name), else_branch);
  if (Tcond) {
    odsState.addAttribute(getTcondAttrName(odsState.name), Tcond);
  }
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_branch;
  ::mlir::Attribute tblgen_Tcond;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'else_branch'");
    if (namedAttrIt->getName() == getElseBranchAttrName()) {
      tblgen_else_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTcondAttrName()) {
      tblgen_Tcond = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTinAttrName()) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getToutAttrName()) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_then_branch;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'then_branch'");
    if (namedAttrIt->getName() == getThenBranchAttrName()) {
      tblgen_then_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_then_branch, "then_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_else_branch, "else_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops17(*this, tblgen_Tcond, "Tcond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult IfOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyIfLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::IfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::IfRegionOp definitions
//===----------------------------------------------------------------------===//

IfRegionOpAdaptor::IfRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.IfRegion", odsAttrs.getContext());
}

IfRegionOpAdaptor::IfRegionOpAdaptor(IfRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IfRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IfRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange IfRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IfRegionOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange IfRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr IfRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DictionaryAttr IfRegionOpAdaptor::getThenAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, IfRegionOp::getThenAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > IfRegionOpAdaptor::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr IfRegionOpAdaptor::getElseAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, IfRegionOp::getElseAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > IfRegionOpAdaptor::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr IfRegionOpAdaptor::getThenRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, IfRegionOp::getThenRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> IfRegionOpAdaptor::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr IfRegionOpAdaptor::getElseRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, IfRegionOp::getElseRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> IfRegionOpAdaptor::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange IfRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &IfRegionOpAdaptor::getThenRegion() {
  return *odsRegions[0];
}

::mlir::Region &IfRegionOpAdaptor::getElseRegion() {
  return *odsRegions[1];
}

::mlir::LogicalResult IfRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_attrs;
  ::mlir::Attribute tblgen_else_region_attrs;
  ::mlir::Attribute tblgen_then_attrs;
  ::mlir::Attribute tblgen_then_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == IfRegionOp::getElseAttrsAttrName(*odsOpName)) {
      tblgen_else_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == IfRegionOp::getElseRegionAttrsAttrName(*odsOpName)) {
      tblgen_else_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == IfRegionOp::getThenAttrsAttrName(*odsOpName)) {
      tblgen_then_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == IfRegionOp::getThenRegionAttrsAttrName(*odsOpName)) {
      tblgen_then_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_then_attrs && !((tblgen_then_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'then_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_else_attrs && !((tblgen_else_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'else_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_then_region_attrs && !((tblgen_then_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'then_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_else_region_attrs && !((tblgen_else_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.IfRegion' op ""attribute 'else_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IfRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range IfRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> IfRegionOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range IfRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange IfRegionOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange IfRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IfRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range IfRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range IfRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> IfRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &IfRegionOp::getThenRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &IfRegionOp::getElseRegion() {
  return (*this)->getRegion(1);
}

::mlir::DictionaryAttr IfRegionOp::getThenAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getThenAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > IfRegionOp::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr IfRegionOp::getElseAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElseAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > IfRegionOp::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr IfRegionOp::getThenRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getThenRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> IfRegionOp::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr IfRegionOp::getElseRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElseRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> IfRegionOp::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void IfRegionOp::setThenAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getThenAttrsAttrName(), attr);
}

void IfRegionOp::setElseAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getElseAttrsAttrName(), attr);
}

void IfRegionOp::setThenRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getThenRegionAttrsAttrName(), attr);
}

void IfRegionOp::setElseRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getElseRegionAttrsAttrName(), attr);
}

::mlir::Attribute IfRegionOp::removeThen_attrsAttr() {
  return (*this)->removeAttr(getThenAttrsAttrName());
}

::mlir::Attribute IfRegionOp::removeElse_attrsAttr() {
  return (*this)->removeAttr(getElseAttrsAttrName());
}

::mlir::Attribute IfRegionOp::removeThen_region_attrsAttr() {
  return (*this)->removeAttr(getThenRegionAttrsAttrName());
}

::mlir::Attribute IfRegionOp::removeElse_region_attrsAttr() {
  return (*this)->removeAttr(getElseRegionAttrsAttrName());
}

void IfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.addAttribute(getThenAttrsAttrName(odsState.name), then_attrs);
  }
  if (else_attrs) {
    odsState.addAttribute(getElseAttrsAttrName(odsState.name), else_attrs);
  }
  if (then_region_attrs) {
    odsState.addAttribute(getThenRegionAttrsAttrName(odsState.name), then_region_attrs);
  }
  if (else_region_attrs) {
    odsState.addAttribute(getElseRegionAttrsAttrName(odsState.name), else_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void IfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.addAttribute(getThenAttrsAttrName(odsState.name), then_attrs);
  }
  if (else_attrs) {
    odsState.addAttribute(getElseAttrsAttrName(odsState.name), else_attrs);
  }
  if (then_region_attrs) {
    odsState.addAttribute(getThenRegionAttrsAttrName(odsState.name), then_region_attrs);
  }
  if (else_region_attrs) {
    odsState.addAttribute(getElseRegionAttrsAttrName(odsState.name), else_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IfRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IfRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_attrs;
  ::mlir::Attribute tblgen_else_region_attrs;
  ::mlir::Attribute tblgen_then_attrs;
  ::mlir::Attribute tblgen_then_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getElseAttrsAttrName()) {
      tblgen_else_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getElseRegionAttrsAttrName()) {
      tblgen_else_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThenAttrsAttrName()) {
      tblgen_then_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThenRegionAttrsAttrName()) {
      tblgen_then_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_then_attrs, "then_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_else_attrs, "else_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_then_region_attrs, "then_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_else_region_attrs, "else_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "then_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "else_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult IfRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult IfRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(condRawOperands);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> then_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> else_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> condTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("then"))
    return ::mlir::failure();

  if (parser.parseRegion(*then_regionRegion))
    return ::mlir::failure();

  if (then_regionRegion->empty()) then_regionRegion->emplaceBlock();
  if (parser.parseKeyword("else"))
    return ::mlir::failure();

  if (parser.parseRegion(*else_regionRegion))
    return ::mlir::failure();

  if (else_regionRegion->empty()) else_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType cond__outs_functionType;
  if (parser.parseType(cond__outs_functionType))
    return ::mlir::failure();
  condTypes = cond__outs_functionType.getInputs();
  outsTypes = cond__outs_functionType.getResults();
  result.addRegion(std::move(then_regionRegion));
  result.addRegion(std::move(else_regionRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IfRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "then";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getThenRegion());
  _odsPrinter << ' ' << "else";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getElseRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getCond().getType()), getOuts().getTypes());
}

StringRef IfRegionOp::getDefaultDialect() { return "tfg"; }
void IfRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp IfRegionOp::getThenYield() {
  return cast<YieldOp>(getThenBlock().getTerminator());
}
YieldOp IfRegionOp::getElseYield() {
  return cast<YieldOp>(getElseBlock().getTerminator());
}

void IfRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpSuccessorRegions(*this, index, operands, regions);
}
void IfRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
}

LogicalResult IfRegionOp::verify() { return VerifyIfLikeRegionOp(*this); }

BlockArgument IfRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument IfRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument IfRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument IfRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::IfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.return", odsAttrs.getContext());
}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ReturnOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ReturnOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ReturnOpAdaptor::getControlRetAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ReturnOp::getControlRetAttrsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr ReturnOpAdaptor::getControlRetAttrs() {
  auto attr = getControlRetAttrsAttr();
  return attr;
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_control_ret_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.return' op ""requires attribute 'control_ret_attrs'");
    if (namedAttrIt->getName() == ReturnOp::getControlRetAttrsAttrName(*odsOpName)) {
      tblgen_control_ret_attrs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_control_ret_attrs && !(((tblgen_control_ret_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_control_ret_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'tfg.return' op ""attribute 'control_ret_attrs' failed to satisfy constraint: an array of dictionaries");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr ReturnOp::getControlRetAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getControlRetAttrsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ReturnOp::getControlRetAttrs() {
  auto attr = getControlRetAttrsAttr();
  return attr;
}

void ReturnOp::setControlRetAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getControlRetAttrsAttrName(), attr);
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::mlir::ArrayAttr control_ret_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getControlRetAttrsAttrName(odsState.name), control_ret_attrs);
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::ArrayAttr control_ret_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getControlRetAttrsAttrName(odsState.name), control_ret_attrs);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_control_ret_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'control_ret_attrs'");
    if (namedAttrIt->getName() == getControlRetAttrsAttrName()) {
      tblgen_control_ret_attrs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_control_ret_attrs, "control_ret_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::ReturnOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulCaseOp definitions
//===----------------------------------------------------------------------===//

StatefulCaseOpAdaptor::StatefulCaseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatefulCase", odsAttrs.getContext());
}

StatefulCaseOpAdaptor::StatefulCaseOpAdaptor(StatefulCaseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatefulCaseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatefulCaseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatefulCaseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatefulCaseOpAdaptor::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatefulCaseOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatefulCaseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr StatefulCaseOpAdaptor::getBranchesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulCaseOp::getBranchesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr StatefulCaseOpAdaptor::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::mlir::ArrayAttr StatefulCaseOpAdaptor::getTinAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatefulCaseOp::getTinAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseOpAdaptor::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulCaseOpAdaptor::getToutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatefulCaseOp::getToutAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseOpAdaptor::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulCaseOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, StatefulCaseOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult StatefulCaseOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branches;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulCase' op ""requires attribute 'branches'");
    if (namedAttrIt->getName() == StatefulCaseOp::getBranchesAttrName(*odsOpName)) {
      tblgen_branches = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatefulCaseOp::getTinAttrName(*odsOpName)) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulCaseOp::getToutAttrName(*odsOpName)) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StatefulCaseOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_branches && !(((tblgen_branches.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_branches.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::FuncAttr>())); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'branches' failed to satisfy constraint: an array of functions");

  if (tblgen_Tin && !(((tblgen_Tin.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tin.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((tblgen_Tout.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tout.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.StatefulCase' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatefulCaseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatefulCaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatefulCaseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatefulCaseOp::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatefulCaseOp::getArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatefulCaseOp::getBranchIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatefulCaseOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulCaseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatefulCaseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatefulCaseOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatefulCaseOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::ArrayAttr StatefulCaseOp::getBranchesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr StatefulCaseOp::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::mlir::ArrayAttr StatefulCaseOp::getTinAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getTinAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulCaseOp::getToutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getToutAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulCaseOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatefulCaseOp::setBranchesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBranchesAttrName(), attr);
}

void StatefulCaseOp::setTinAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTinAttrName(), attr);
}

void StatefulCaseOp::setToutAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getToutAttrName(), attr);
}

void StatefulCaseOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute StatefulCaseOp::removeTinAttr() {
  return (*this)->removeAttr(getTinAttrName());
}

::mlir::Attribute StatefulCaseOp::removeToutAttr() {
  return (*this)->removeAttr(getToutAttrName());
}

::mlir::Attribute StatefulCaseOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void StatefulCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches) {
      build(odsBuilder, odsState, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatefulCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.addAttribute(getBranchesAttrName(odsState.name), branches);
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.addAttribute(getBranchesAttrName(odsState.name), branches);
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulCaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatefulCaseOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branches;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'branches'");
    if (namedAttrIt->getName() == getBranchesAttrName()) {
      tblgen_branches = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTinAttrName()) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getToutAttrName()) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops0(*this, tblgen_branches, "branches")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulCaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatefulCaseOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyCaseLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulCaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulCaseRegionOp definitions
//===----------------------------------------------------------------------===//

StatefulCaseRegionOpAdaptor::StatefulCaseRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatefulCaseRegion", odsAttrs.getContext());
}

StatefulCaseRegionOpAdaptor::StatefulCaseRegionOpAdaptor(StatefulCaseRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatefulCaseRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatefulCaseRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatefulCaseRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatefulCaseRegionOpAdaptor::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatefulCaseRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatefulCaseRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr StatefulCaseRegionOpAdaptor::getBranchAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulCaseRegionOp::getBranchAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseRegionOpAdaptor::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulCaseRegionOpAdaptor::getRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulCaseRegionOp::getRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseRegionOpAdaptor::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::RegionRange StatefulCaseRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::RegionRange StatefulCaseRegionOpAdaptor::getBranches() {
  return odsRegions.drop_front(0);
}

::mlir::LogicalResult StatefulCaseRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_attrs;
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StatefulCaseRegionOp::getBranchAttrsAttrName(*odsOpName)) {
      tblgen_branch_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulCaseRegionOp::getRegionAttrsAttrName(*odsOpName)) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_branch_attrs && !(((tblgen_branch_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_branch_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'tfg.StatefulCaseRegion' op ""attribute 'branch_attrs' failed to satisfy constraint: an array of dictionaries");

  if (tblgen_region_attrs && !(((tblgen_region_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_region_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tfg::RegionAttr>())); }))))
    return emitError(loc, "'tfg.StatefulCaseRegion' op ""attribute 'region_attrs' failed to satisfy constraint: case region preserved attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatefulCaseRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatefulCaseRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> StatefulCaseRegionOp::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatefulCaseRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatefulCaseRegionOp::getBranchIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatefulCaseRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulCaseRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatefulCaseRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatefulCaseRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatefulCaseRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::MutableArrayRef<::mlir::Region> StatefulCaseRegionOp::getBranches() {
  return (*this)->getRegions().drop_front(0);
}

::mlir::ArrayAttr StatefulCaseRegionOp::getBranchAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseRegionOp::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulCaseRegionOp::getRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getRegionAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulCaseRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatefulCaseRegionOp::setBranchAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBranchAttrsAttrName(), attr);
}

void StatefulCaseRegionOp::setRegionAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getRegionAttrsAttrName(), attr);
}

::mlir::Attribute StatefulCaseRegionOp::removeBranch_attrsAttr() {
  return (*this)->removeAttr(getBranchAttrsAttrName());
}

::mlir::Attribute StatefulCaseRegionOp::removeRegion_attrsAttr() {
  return (*this)->removeAttr(getRegionAttrsAttrName());
}

void StatefulCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.addAttribute(getBranchAttrsAttrName(odsState.name), branch_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.addAttribute(getBranchAttrsAttrName(odsState.name), branch_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulCaseRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatefulCaseRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_attrs;
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBranchAttrsAttrName()) {
      tblgen_branch_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getRegionAttrsAttrName()) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_branch_attrs, "branch_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops4(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getBranches())
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "branches", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulCaseRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatefulCaseRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand branch_indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> branch_indexOperands(branch_indexRawOperands);  ::llvm::SMLoc branch_indexOperandsLoc;
  (void)branch_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> branchesRegions;
  ::llvm::ArrayRef<::mlir::Type> branch_indexTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  branch_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(branch_indexRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      branchesRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        branchesRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : branchesRegions)
    if (region->empty()) region->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType branch_index__outs_functionType;
  if (parser.parseType(branch_index__outs_functionType))
    return ::mlir::failure();
  branch_indexTypes = branch_index__outs_functionType.getInputs();
  outsTypes = branch_index__outs_functionType.getResults();
  result.addRegions(branchesRegions);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(branch_indexOperands, branch_indexTypes, branch_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulCaseRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBranchIndex();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
    llvm::interleaveComma(getBranches(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getBranchIndex().getType()), getOuts().getTypes());
}

StringRef StatefulCaseRegionOp::getDefaultDialect() { return "tfg"; }
void StatefulCaseRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatefulCaseRegionOp::branch_yield(unsigned idx) {
  return cast<YieldOp>(getBranchBlock(idx).getTerminator());
}

void StatefulCaseRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpSuccessorRegions(*this, index, operands, regions);
}
void StatefulCaseRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
}

LogicalResult StatefulCaseRegionOp::verify() { return VerifyCaseLikeRegionOp(*this); }

BlockArgument StatefulCaseRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatefulCaseRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatefulCaseRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatefulCaseRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulCaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulIfOp definitions
//===----------------------------------------------------------------------===//

StatefulIfOpAdaptor::StatefulIfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatefulIf", odsAttrs.getContext());
}

StatefulIfOpAdaptor::StatefulIfOpAdaptor(StatefulIfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatefulIfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatefulIfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatefulIfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatefulIfOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatefulIfOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatefulIfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr StatefulIfOpAdaptor::getThenBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, StatefulIfOp::getThenBranchAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatefulIfOpAdaptor::getThenBranch() {
  auto attr = getThenBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulIfOpAdaptor::getElseBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatefulIfOp::getElseBranchAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatefulIfOpAdaptor::getElseBranch() {
  auto attr = getElseBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::TypeAttr StatefulIfOpAdaptor::getTcondAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulIfOp::getTcondAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::llvm::Optional<::mlir::Type> StatefulIfOpAdaptor::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::ArrayAttr StatefulIfOpAdaptor::getTinAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulIfOp::getTinAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulIfOpAdaptor::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulIfOpAdaptor::getToutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulIfOp::getToutAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulIfOpAdaptor::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulIfOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, StatefulIfOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulIfOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult StatefulIfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_branch;
  ::mlir::Attribute tblgen_Tcond;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulIf' op ""requires attribute 'else_branch'");
    if (namedAttrIt->getName() == StatefulIfOp::getElseBranchAttrName(*odsOpName)) {
      tblgen_else_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatefulIfOp::getTcondAttrName(*odsOpName)) {
      tblgen_Tcond = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulIfOp::getTinAttrName(*odsOpName)) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulIfOp::getToutAttrName(*odsOpName)) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_then_branch;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulIf' op ""requires attribute 'then_branch'");
    if (namedAttrIt->getName() == StatefulIfOp::getThenBranchAttrName(*odsOpName)) {
      tblgen_then_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatefulIfOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_then_branch && !((tblgen_then_branch.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'then_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_else_branch && !((tblgen_else_branch.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'else_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_Tcond && !(((tblgen_Tcond.isa<::mlir::TypeAttr>())) && ((tblgen_Tcond.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'Tcond' failed to satisfy constraint: any type attribute");

  if (tblgen_Tin && !(((tblgen_Tin.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tin.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((tblgen_Tout.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tout.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.StatefulIf' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatefulIfOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatefulIfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatefulIfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatefulIfOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatefulIfOp::getArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatefulIfOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatefulIfOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulIfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatefulIfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatefulIfOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatefulIfOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr StatefulIfOp::getThenBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getThenBranchAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulIfOp::getThenBranch() {
  auto attr = getThenBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulIfOp::getElseBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getElseBranchAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulIfOp::getElseBranch() {
  auto attr = getElseBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::TypeAttr StatefulIfOp::getTcondAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getTcondAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> StatefulIfOp::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::ArrayAttr StatefulIfOp::getTinAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getTinAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulIfOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulIfOp::getToutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getToutAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulIfOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulIfOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulIfOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatefulIfOp::setThenBranchAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getThenBranchAttrName(), attr);
}

void StatefulIfOp::setElseBranchAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getElseBranchAttrName(), attr);
}

void StatefulIfOp::setTcondAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTcondAttrName(), attr);
}

void StatefulIfOp::setTinAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTinAttrName(), attr);
}

void StatefulIfOp::setToutAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getToutAttrName(), attr);
}

void StatefulIfOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute StatefulIfOp::removeTcondAttr() {
  return (*this)->removeAttr(getTcondAttrName());
}

::mlir::Attribute StatefulIfOp::removeTinAttr() {
  return (*this)->removeAttr(getTinAttrName());
}

::mlir::Attribute StatefulIfOp::removeToutAttr() {
  return (*this)->removeAttr(getToutAttrName());
}

::mlir::Attribute StatefulIfOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void StatefulIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch) {
      build(odsBuilder, odsState, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatefulIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addAttribute(getThenBranchAttrName(odsState.name), then_branch);
  odsState.addAttribute(getElseBranchAttrName(odsState.name), else_branch);
  if (Tcond) {
    odsState.addAttribute(getTcondAttrName(odsState.name), Tcond);
  }
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addAttribute(getThenBranchAttrName(odsState.name), then_branch);
  odsState.addAttribute(getElseBranchAttrName(odsState.name), else_branch);
  if (Tcond) {
    odsState.addAttribute(getTcondAttrName(odsState.name), Tcond);
  }
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulIfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatefulIfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_branch;
  ::mlir::Attribute tblgen_Tcond;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'else_branch'");
    if (namedAttrIt->getName() == getElseBranchAttrName()) {
      tblgen_else_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTcondAttrName()) {
      tblgen_Tcond = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTinAttrName()) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getToutAttrName()) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_then_branch;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'then_branch'");
    if (namedAttrIt->getName() == getThenBranchAttrName()) {
      tblgen_then_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_then_branch, "then_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_else_branch, "else_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops17(*this, tblgen_Tcond, "Tcond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulIfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatefulIfOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyIfLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulIfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulIfRegionOp definitions
//===----------------------------------------------------------------------===//

StatefulIfRegionOpAdaptor::StatefulIfRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatefulIfRegion", odsAttrs.getContext());
}

StatefulIfRegionOpAdaptor::StatefulIfRegionOpAdaptor(StatefulIfRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatefulIfRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatefulIfRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatefulIfRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatefulIfRegionOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatefulIfRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatefulIfRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DictionaryAttr StatefulIfRegionOpAdaptor::getThenAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulIfRegionOp::getThenAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulIfRegionOpAdaptor::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatefulIfRegionOpAdaptor::getElseAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulIfRegionOp::getElseAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulIfRegionOpAdaptor::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulIfRegionOpAdaptor::getThenRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulIfRegionOp::getThenRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulIfRegionOpAdaptor::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulIfRegionOpAdaptor::getElseRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatefulIfRegionOp::getElseRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulIfRegionOpAdaptor::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange StatefulIfRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &StatefulIfRegionOpAdaptor::getThenRegion() {
  return *odsRegions[0];
}

::mlir::Region &StatefulIfRegionOpAdaptor::getElseRegion() {
  return *odsRegions[1];
}

::mlir::LogicalResult StatefulIfRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_attrs;
  ::mlir::Attribute tblgen_else_region_attrs;
  ::mlir::Attribute tblgen_then_attrs;
  ::mlir::Attribute tblgen_then_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StatefulIfRegionOp::getElseAttrsAttrName(*odsOpName)) {
      tblgen_else_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulIfRegionOp::getElseRegionAttrsAttrName(*odsOpName)) {
      tblgen_else_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulIfRegionOp::getThenAttrsAttrName(*odsOpName)) {
      tblgen_then_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulIfRegionOp::getThenRegionAttrsAttrName(*odsOpName)) {
      tblgen_then_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_then_attrs && !((tblgen_then_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'then_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_else_attrs && !((tblgen_else_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'else_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_then_region_attrs && !((tblgen_then_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'then_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_else_region_attrs && !((tblgen_else_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatefulIfRegion' op ""attribute 'else_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatefulIfRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatefulIfRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> StatefulIfRegionOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatefulIfRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatefulIfRegionOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatefulIfRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulIfRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatefulIfRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatefulIfRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatefulIfRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &StatefulIfRegionOp::getThenRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &StatefulIfRegionOp::getElseRegion() {
  return (*this)->getRegion(1);
}

::mlir::DictionaryAttr StatefulIfRegionOp::getThenAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getThenAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulIfRegionOp::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatefulIfRegionOp::getElseAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElseAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulIfRegionOp::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulIfRegionOp::getThenRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getThenRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulIfRegionOp::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulIfRegionOp::getElseRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElseRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulIfRegionOp::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void StatefulIfRegionOp::setThenAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getThenAttrsAttrName(), attr);
}

void StatefulIfRegionOp::setElseAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getElseAttrsAttrName(), attr);
}

void StatefulIfRegionOp::setThenRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getThenRegionAttrsAttrName(), attr);
}

void StatefulIfRegionOp::setElseRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getElseRegionAttrsAttrName(), attr);
}

::mlir::Attribute StatefulIfRegionOp::removeThen_attrsAttr() {
  return (*this)->removeAttr(getThenAttrsAttrName());
}

::mlir::Attribute StatefulIfRegionOp::removeElse_attrsAttr() {
  return (*this)->removeAttr(getElseAttrsAttrName());
}

::mlir::Attribute StatefulIfRegionOp::removeThen_region_attrsAttr() {
  return (*this)->removeAttr(getThenRegionAttrsAttrName());
}

::mlir::Attribute StatefulIfRegionOp::removeElse_region_attrsAttr() {
  return (*this)->removeAttr(getElseRegionAttrsAttrName());
}

void StatefulIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.addAttribute(getThenAttrsAttrName(odsState.name), then_attrs);
  }
  if (else_attrs) {
    odsState.addAttribute(getElseAttrsAttrName(odsState.name), else_attrs);
  }
  if (then_region_attrs) {
    odsState.addAttribute(getThenRegionAttrsAttrName(odsState.name), then_region_attrs);
  }
  if (else_region_attrs) {
    odsState.addAttribute(getElseRegionAttrsAttrName(odsState.name), else_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.addAttribute(getThenAttrsAttrName(odsState.name), then_attrs);
  }
  if (else_attrs) {
    odsState.addAttribute(getElseAttrsAttrName(odsState.name), else_attrs);
  }
  if (then_region_attrs) {
    odsState.addAttribute(getThenRegionAttrsAttrName(odsState.name), then_region_attrs);
  }
  if (else_region_attrs) {
    odsState.addAttribute(getElseRegionAttrsAttrName(odsState.name), else_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulIfRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatefulIfRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_attrs;
  ::mlir::Attribute tblgen_else_region_attrs;
  ::mlir::Attribute tblgen_then_attrs;
  ::mlir::Attribute tblgen_then_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getElseAttrsAttrName()) {
      tblgen_else_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getElseRegionAttrsAttrName()) {
      tblgen_else_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThenAttrsAttrName()) {
      tblgen_then_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThenRegionAttrsAttrName()) {
      tblgen_then_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_then_attrs, "then_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_else_attrs, "else_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_then_region_attrs, "then_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_else_region_attrs, "else_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "then_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "else_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulIfRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatefulIfRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(condRawOperands);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> then_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> else_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> condTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("then"))
    return ::mlir::failure();

  if (parser.parseRegion(*then_regionRegion))
    return ::mlir::failure();

  if (then_regionRegion->empty()) then_regionRegion->emplaceBlock();
  if (parser.parseKeyword("else"))
    return ::mlir::failure();

  if (parser.parseRegion(*else_regionRegion))
    return ::mlir::failure();

  if (else_regionRegion->empty()) else_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType cond__outs_functionType;
  if (parser.parseType(cond__outs_functionType))
    return ::mlir::failure();
  condTypes = cond__outs_functionType.getInputs();
  outsTypes = cond__outs_functionType.getResults();
  result.addRegion(std::move(then_regionRegion));
  result.addRegion(std::move(else_regionRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulIfRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "then";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getThenRegion());
  _odsPrinter << ' ' << "else";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getElseRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getCond().getType()), getOuts().getTypes());
}

StringRef StatefulIfRegionOp::getDefaultDialect() { return "tfg"; }
void StatefulIfRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatefulIfRegionOp::getThenYield() {
  return cast<YieldOp>(getThenBlock().getTerminator());
}
YieldOp StatefulIfRegionOp::getElseYield() {
  return cast<YieldOp>(getElseBlock().getTerminator());
}

void StatefulIfRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpSuccessorRegions(*this, index, operands, regions);
}
void StatefulIfRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
}

LogicalResult StatefulIfRegionOp::verify() { return VerifyIfLikeRegionOp(*this); }

BlockArgument StatefulIfRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatefulIfRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatefulIfRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatefulIfRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulIfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulWhileOp definitions
//===----------------------------------------------------------------------===//

StatefulWhileOpAdaptor::StatefulWhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatefulWhile", odsAttrs.getContext());
}

StatefulWhileOpAdaptor::StatefulWhileOpAdaptor(StatefulWhileOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatefulWhileOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatefulWhileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatefulWhileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange StatefulWhileOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr StatefulWhileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr StatefulWhileOpAdaptor::getCondAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, StatefulWhileOp::getCondAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatefulWhileOpAdaptor::getCond() {
  auto attr = getCondAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulWhileOpAdaptor::getBodyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulWhileOp::getBodyAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatefulWhileOpAdaptor::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::IntegerAttr StatefulWhileOpAdaptor::getParallelIterationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, StatefulWhileOp::getParallelIterationsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t StatefulWhileOpAdaptor::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr StatefulWhileOpAdaptor::getTAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, StatefulWhileOp::getTAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulWhileOpAdaptor::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulWhileOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, StatefulWhileOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulWhileOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult StatefulWhileOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulWhile' op ""requires attribute 'body'");
    if (namedAttrIt->getName() == StatefulWhileOp::getBodyAttrName(*odsOpName)) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatefulWhileOp::getTAttrName(*odsOpName)) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_cond;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulWhile' op ""requires attribute 'cond'");
    if (namedAttrIt->getName() == StatefulWhileOp::getCondAttrName(*odsOpName)) {
      tblgen_cond = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulWhile' op ""requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == StatefulWhileOp::getParallelIterationsAttrName(*odsOpName)) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatefulWhileOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_cond && !((tblgen_cond.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'cond' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_body && !((tblgen_body.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_parallel_iterations && !(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_T && !(((tblgen_T.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_T.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'T' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.StatefulWhile' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatefulWhileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatefulWhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatefulWhileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range StatefulWhileOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange StatefulWhileOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulWhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatefulWhileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatefulWhileOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatefulWhileOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr StatefulWhileOp::getCondAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getCondAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulWhileOp::getCond() {
  auto attr = getCondAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulWhileOp::getBodyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatefulWhileOp::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::IntegerAttr StatefulWhileOp::getParallelIterationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getParallelIterationsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t StatefulWhileOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr StatefulWhileOp::getTAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getTAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulWhileOp::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatefulWhileOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatefulWhileOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatefulWhileOp::setCondAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getCondAttrName(), attr);
}

void StatefulWhileOp::setBodyAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getBodyAttrName(), attr);
}

void StatefulWhileOp::setParallelIterationsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getParallelIterationsAttrName(), attr);
}

void StatefulWhileOp::setTAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTAttrName(), attr);
}

void StatefulWhileOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute StatefulWhileOp::removeTAttr() {
  return (*this)->removeAttr(getTAttrName());
}

::mlir::Attribute StatefulWhileOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations) {
      build(odsBuilder, odsState, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatefulWhileOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'body'");
    if (namedAttrIt->getName() == getBodyAttrName()) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTAttrName()) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_cond;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cond'");
    if (namedAttrIt->getName() == getCondAttrName()) {
      tblgen_cond = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == getParallelIterationsAttrName()) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_cond, "cond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulWhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatefulWhileOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyWhileLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulWhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulWhileRegionOp definitions
//===----------------------------------------------------------------------===//

StatefulWhileRegionOpAdaptor::StatefulWhileRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatefulWhileRegion", odsAttrs.getContext());
}

StatefulWhileRegionOpAdaptor::StatefulWhileRegionOpAdaptor(StatefulWhileRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatefulWhileRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatefulWhileRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatefulWhileRegionOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange StatefulWhileRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange StatefulWhileRegionOpAdaptor::getInit() {
  return getODSOperands(0);
}

::mlir::ValueRange StatefulWhileRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatefulWhileRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr StatefulWhileRegionOpAdaptor::getParallelIterationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, StatefulWhileRegionOp::getParallelIterationsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t StatefulWhileRegionOpAdaptor::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr StatefulWhileRegionOpAdaptor::getCondAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulWhileRegionOp::getCondAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulWhileRegionOpAdaptor::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatefulWhileRegionOpAdaptor::getBodyAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulWhileRegionOp::getBodyAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulWhileRegionOpAdaptor::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulWhileRegionOpAdaptor::getCondRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulWhileRegionOp::getCondRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOpAdaptor::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulWhileRegionOpAdaptor::getBodyRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatefulWhileRegionOp::getBodyRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOpAdaptor::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange StatefulWhileRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &StatefulWhileRegionOpAdaptor::getCondRegion() {
  return *odsRegions[0];
}

::mlir::Region &StatefulWhileRegionOpAdaptor::getBodyRegion() {
  return *odsRegions[1];
}

::mlir::LogicalResult StatefulWhileRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  ::mlir::Attribute tblgen_body_region_attrs;
  ::mlir::Attribute tblgen_cond_attrs;
  ::mlir::Attribute tblgen_cond_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulWhileRegion' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == StatefulWhileRegionOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatefulWhileRegionOp::getBodyAttrsAttrName(*odsOpName)) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulWhileRegionOp::getBodyRegionAttrsAttrName(*odsOpName)) {
      tblgen_body_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulWhileRegionOp::getCondAttrsAttrName(*odsOpName)) {
      tblgen_cond_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatefulWhileRegionOp::getCondRegionAttrsAttrName(*odsOpName)) {
      tblgen_cond_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatefulWhileRegion' op ""requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == StatefulWhileRegionOp::getParallelIterationsAttrName(*odsOpName)) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'tfg.StatefulWhileRegion' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_parallel_iterations && !(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_cond_attrs && !((tblgen_cond_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'cond_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_body_attrs && !((tblgen_body_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_cond_region_attrs && !((tblgen_cond_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'cond_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_body_region_attrs && !((tblgen_body_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatefulWhileRegion' op ""attribute 'body_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatefulWhileRegionOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range StatefulWhileRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range StatefulWhileRegionOp::getInit() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range StatefulWhileRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatefulWhileRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange StatefulWhileRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> StatefulWhileRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatefulWhileRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatefulWhileRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatefulWhileRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &StatefulWhileRegionOp::getCondRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &StatefulWhileRegionOp::getBodyRegion() {
  return (*this)->getRegion(1);
}

::mlir::IntegerAttr StatefulWhileRegionOp::getParallelIterationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getParallelIterationsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t StatefulWhileRegionOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr StatefulWhileRegionOp::getCondAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCondAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulWhileRegionOp::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatefulWhileRegionOp::getBodyAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatefulWhileRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulWhileRegionOp::getCondRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCondRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOp::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatefulWhileRegionOp::getBodyRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatefulWhileRegionOp::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void StatefulWhileRegionOp::setParallelIterationsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getParallelIterationsAttrName(), attr);
}

void StatefulWhileRegionOp::setCondAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getCondAttrsAttrName(), attr);
}

void StatefulWhileRegionOp::setBodyAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getBodyAttrsAttrName(), attr);
}

void StatefulWhileRegionOp::setCondRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getCondRegionAttrsAttrName(), attr);
}

void StatefulWhileRegionOp::setBodyRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getBodyRegionAttrsAttrName(), attr);
}

::mlir::Attribute StatefulWhileRegionOp::removeCond_attrsAttr() {
  return (*this)->removeAttr(getCondAttrsAttrName());
}

::mlir::Attribute StatefulWhileRegionOp::removeBody_attrsAttr() {
  return (*this)->removeAttr(getBodyAttrsAttrName());
}

::mlir::Attribute StatefulWhileRegionOp::removeCond_region_attrsAttr() {
  return (*this)->removeAttr(getCondRegionAttrsAttrName());
}

::mlir::Attribute StatefulWhileRegionOp::removeBody_region_attrsAttr() {
  return (*this)->removeAttr(getBodyRegionAttrsAttrName());
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatefulWhileRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatefulWhileRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  ::mlir::Attribute tblgen_body_region_attrs;
  ::mlir::Attribute tblgen_cond_attrs;
  ::mlir::Attribute tblgen_cond_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBodyAttrsAttrName()) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getBodyRegionAttrsAttrName()) {
      tblgen_body_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCondAttrsAttrName()) {
      tblgen_cond_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCondRegionAttrsAttrName()) {
      tblgen_cond_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == getParallelIterationsAttrName()) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_cond_attrs, "cond_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_cond_region_attrs, "cond_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_body_region_attrs, "body_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "cond_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatefulWhileRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatefulWhileRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> cond_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> initTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*cond_regionRegion))
    return ::mlir::failure();

  if (cond_regionRegion->empty()) cond_regionRegion->emplaceBlock();
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  ::mlir::FunctionType init__outs_functionType;
  if (parser.parseType(init__outs_functionType))
    return ::mlir::failure();
  initTypes = init__outs_functionType.getInputs();
  outsTypes = init__outs_functionType.getResults();
  }
  result.addRegion(std::move(cond_regionRegion));
  result.addRegion(std::move(body_regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatefulWhileRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getCondRegion());
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  if (!getInit().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printFunctionalType(getInit().getTypes(), getOuts().getTypes());
  }
}

StringRef StatefulWhileRegionOp::getDefaultDialect() { return "tfg"; }
void StatefulWhileRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

ConditionOp StatefulWhileRegionOp::getCondCondition() {
  return cast<ConditionOp>(getCondBlock().getTerminator());
}
YieldOp StatefulWhileRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}

OperandRange StatefulWhileRegionOp::getSuccessorEntryOperands(
    Optional<unsigned> index) {
  return getInit();
}
void StatefulWhileRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return GetWhileLikeRegionOpSuccessorRegions(*this, index, operands,
                                              regions);
}

LogicalResult StatefulWhileRegionOp::verify() { return VerifyWhileLikeRegionOp(*this); }

BlockArgument StatefulWhileRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatefulWhileRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatefulWhileRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatefulWhileRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulWhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessCaseOp definitions
//===----------------------------------------------------------------------===//

StatelessCaseOpAdaptor::StatelessCaseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatelessCase", odsAttrs.getContext());
}

StatelessCaseOpAdaptor::StatelessCaseOpAdaptor(StatelessCaseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatelessCaseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatelessCaseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatelessCaseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatelessCaseOpAdaptor::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatelessCaseOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatelessCaseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr StatelessCaseOpAdaptor::getBranchesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessCaseOp::getBranchesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr StatelessCaseOpAdaptor::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::mlir::ArrayAttr StatelessCaseOpAdaptor::getTinAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatelessCaseOp::getTinAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseOpAdaptor::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessCaseOpAdaptor::getToutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatelessCaseOp::getToutAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseOpAdaptor::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessCaseOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, StatelessCaseOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult StatelessCaseOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branches;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessCase' op ""requires attribute 'branches'");
    if (namedAttrIt->getName() == StatelessCaseOp::getBranchesAttrName(*odsOpName)) {
      tblgen_branches = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatelessCaseOp::getTinAttrName(*odsOpName)) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessCaseOp::getToutAttrName(*odsOpName)) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StatelessCaseOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_branches && !(((tblgen_branches.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_branches.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::FuncAttr>())); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'branches' failed to satisfy constraint: an array of functions");

  if (tblgen_Tin && !(((tblgen_Tin.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tin.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((tblgen_Tout.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tout.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.StatelessCase' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatelessCaseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatelessCaseOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatelessCaseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatelessCaseOp::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatelessCaseOp::getArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatelessCaseOp::getBranchIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatelessCaseOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessCaseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatelessCaseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatelessCaseOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatelessCaseOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::ArrayAttr StatelessCaseOp::getBranchesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr StatelessCaseOp::getBranches() {
  auto attr = getBranchesAttr();
  return attr;
}

::mlir::ArrayAttr StatelessCaseOp::getTinAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getTinAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessCaseOp::getToutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getToutAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessCaseOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatelessCaseOp::setBranchesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBranchesAttrName(), attr);
}

void StatelessCaseOp::setTinAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTinAttrName(), attr);
}

void StatelessCaseOp::setToutAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getToutAttrName(), attr);
}

void StatelessCaseOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute StatelessCaseOp::removeTinAttr() {
  return (*this)->removeAttr(getTinAttrName());
}

::mlir::Attribute StatelessCaseOp::removeToutAttr() {
  return (*this)->removeAttr(getToutAttrName());
}

::mlir::Attribute StatelessCaseOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void StatelessCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches) {
      build(odsBuilder, odsState, results, branch_index, args, branches,
            /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatelessCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.addAttribute(getBranchesAttrName(odsState.name), branches);
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessCaseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(branch_index);
  odsState.addOperands(args);
  odsState.addAttribute(getBranchesAttrName(odsState.name), branches);
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessCaseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatelessCaseOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branches;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'branches'");
    if (namedAttrIt->getName() == getBranchesAttrName()) {
      tblgen_branches = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTinAttrName()) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getToutAttrName()) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops0(*this, tblgen_branches, "branches")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessCaseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatelessCaseOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyCaseLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessCaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessCaseRegionOp definitions
//===----------------------------------------------------------------------===//

StatelessCaseRegionOpAdaptor::StatelessCaseRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatelessCaseRegion", odsAttrs.getContext());
}

StatelessCaseRegionOpAdaptor::StatelessCaseRegionOpAdaptor(StatelessCaseRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatelessCaseRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatelessCaseRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatelessCaseRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatelessCaseRegionOpAdaptor::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatelessCaseRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatelessCaseRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr StatelessCaseRegionOpAdaptor::getBranchAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessCaseRegionOp::getBranchAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseRegionOpAdaptor::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessCaseRegionOpAdaptor::getRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessCaseRegionOp::getRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseRegionOpAdaptor::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::RegionRange StatelessCaseRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::RegionRange StatelessCaseRegionOpAdaptor::getBranches() {
  return odsRegions.drop_front(0);
}

::mlir::LogicalResult StatelessCaseRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_attrs;
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StatelessCaseRegionOp::getBranchAttrsAttrName(*odsOpName)) {
      tblgen_branch_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessCaseRegionOp::getRegionAttrsAttrName(*odsOpName)) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_branch_attrs && !(((tblgen_branch_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_branch_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::DictionaryAttr>())); }))))
    return emitError(loc, "'tfg.StatelessCaseRegion' op ""attribute 'branch_attrs' failed to satisfy constraint: an array of dictionaries");

  if (tblgen_region_attrs && !(((tblgen_region_attrs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_region_attrs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tfg::RegionAttr>())); }))))
    return emitError(loc, "'tfg.StatelessCaseRegion' op ""attribute 'region_attrs' failed to satisfy constraint: case region preserved attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatelessCaseRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatelessCaseRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> StatelessCaseRegionOp::getBranchIndex() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatelessCaseRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatelessCaseRegionOp::getBranchIndexMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatelessCaseRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessCaseRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatelessCaseRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatelessCaseRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatelessCaseRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::MutableArrayRef<::mlir::Region> StatelessCaseRegionOp::getBranches() {
  return (*this)->getRegions().drop_front(0);
}

::mlir::ArrayAttr StatelessCaseRegionOp::getBranchAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBranchAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseRegionOp::getBranchAttrs() {
  auto attr = getBranchAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessCaseRegionOp::getRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getRegionAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessCaseRegionOp::getRegionAttrs() {
  auto attr = getRegionAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatelessCaseRegionOp::setBranchAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getBranchAttrsAttrName(), attr);
}

void StatelessCaseRegionOp::setRegionAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getRegionAttrsAttrName(), attr);
}

::mlir::Attribute StatelessCaseRegionOp::removeBranch_attrsAttr() {
  return (*this)->removeAttr(getBranchAttrsAttrName());
}

::mlir::Attribute StatelessCaseRegionOp::removeRegion_attrsAttr() {
  return (*this)->removeAttr(getRegionAttrsAttrName());
}

void StatelessCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.addAttribute(getBranchAttrsAttrName(odsState.name), branch_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessCaseRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount) {
  odsState.addOperands(branch_index);
  odsState.addOperands(ctls);
  if (branch_attrs) {
    odsState.addAttribute(getBranchAttrsAttrName(odsState.name), branch_attrs);
  }
  if (region_attrs) {
    odsState.addAttribute(getRegionAttrsAttrName(odsState.name), region_attrs);
  }
  for (unsigned i = 0; i < branchesCount; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessCaseRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != numRegions; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatelessCaseRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_branch_attrs;
  ::mlir::Attribute tblgen_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getBranchAttrsAttrName()) {
      tblgen_branch_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getRegionAttrsAttrName()) {
      tblgen_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops3(*this, tblgen_branch_attrs, "branch_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops4(*this, tblgen_region_attrs, "region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : getBranches())
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "branches", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessCaseRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatelessCaseRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand branch_indexRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> branch_indexOperands(branch_indexRawOperands);  ::llvm::SMLoc branch_indexOperandsLoc;
  (void)branch_indexOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<std::unique_ptr<::mlir::Region>, 2> branchesRegions;
  ::llvm::ArrayRef<::mlir::Type> branch_indexTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  branch_indexOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(branch_indexRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  {
    std::unique_ptr<::mlir::Region> region;
    auto firstRegionResult = parser.parseOptionalRegion(region);
    if (firstRegionResult.has_value()) {
      if (failed(*firstRegionResult))
        return ::mlir::failure();
      branchesRegions.emplace_back(std::move(region));

      // Parse any trailing regions.
      while (succeeded(parser.parseOptionalComma())) {
        region = std::make_unique<::mlir::Region>();
        if (parser.parseRegion(*region))
          return ::mlir::failure();
        branchesRegions.emplace_back(std::move(region));
      }
    }
  }

  for (auto &region : branchesRegions)
    if (region->empty()) region->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType branch_index__outs_functionType;
  if (parser.parseType(branch_index__outs_functionType))
    return ::mlir::failure();
  branch_indexTypes = branch_index__outs_functionType.getInputs();
  outsTypes = branch_index__outs_functionType.getResults();
  result.addRegions(branchesRegions);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(branch_indexOperands, branch_indexTypes, branch_indexOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessCaseRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBranchIndex();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
    llvm::interleaveComma(getBranches(), _odsPrinter, [&](::mlir::Region &region) {
        _odsPrinter.printRegion(region);
    });
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getBranchIndex().getType()), getOuts().getTypes());
}

StringRef StatelessCaseRegionOp::getDefaultDialect() { return "tfg"; }
void StatelessCaseRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatelessCaseRegionOp::branch_yield(unsigned idx) {
  return cast<YieldOp>(getBranchBlock(idx).getTerminator());
}

void StatelessCaseRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetCaseLikeRegionOpSuccessorRegions(*this, index, operands, regions);
}
void StatelessCaseRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(getNumRegions(), /*Elt=*/{0, 1});
}

LogicalResult StatelessCaseRegionOp::verify() { return VerifyCaseLikeRegionOp(*this); }

BlockArgument StatelessCaseRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatelessCaseRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatelessCaseRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatelessCaseRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessCaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessIfOp definitions
//===----------------------------------------------------------------------===//

StatelessIfOpAdaptor::StatelessIfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatelessIf", odsAttrs.getContext());
}

StatelessIfOpAdaptor::StatelessIfOpAdaptor(StatelessIfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatelessIfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatelessIfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatelessIfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatelessIfOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatelessIfOpAdaptor::getArgs() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatelessIfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr StatelessIfOpAdaptor::getThenBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, StatelessIfOp::getThenBranchAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatelessIfOpAdaptor::getThenBranch() {
  auto attr = getThenBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessIfOpAdaptor::getElseBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatelessIfOp::getElseBranchAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatelessIfOpAdaptor::getElseBranch() {
  auto attr = getElseBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::TypeAttr StatelessIfOpAdaptor::getTcondAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessIfOp::getTcondAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
  return attr;
}

::llvm::Optional<::mlir::Type> StatelessIfOpAdaptor::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::ArrayAttr StatelessIfOpAdaptor::getTinAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessIfOp::getTinAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessIfOpAdaptor::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessIfOpAdaptor::getToutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessIfOp::getToutAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessIfOpAdaptor::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessIfOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, StatelessIfOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessIfOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult StatelessIfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_branch;
  ::mlir::Attribute tblgen_Tcond;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessIf' op ""requires attribute 'else_branch'");
    if (namedAttrIt->getName() == StatelessIfOp::getElseBranchAttrName(*odsOpName)) {
      tblgen_else_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatelessIfOp::getTcondAttrName(*odsOpName)) {
      tblgen_Tcond = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessIfOp::getTinAttrName(*odsOpName)) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessIfOp::getToutAttrName(*odsOpName)) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_then_branch;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessIf' op ""requires attribute 'then_branch'");
    if (namedAttrIt->getName() == StatelessIfOp::getThenBranchAttrName(*odsOpName)) {
      tblgen_then_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatelessIfOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_then_branch && !((tblgen_then_branch.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'then_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_else_branch && !((tblgen_else_branch.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'else_branch' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_Tcond && !(((tblgen_Tcond.isa<::mlir::TypeAttr>())) && ((tblgen_Tcond.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'Tcond' failed to satisfy constraint: any type attribute");

  if (tblgen_Tin && !(((tblgen_Tin.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tin.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'Tin' failed to satisfy constraint: type array attribute");

  if (tblgen_Tout && !(((tblgen_Tout.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_Tout.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'Tout' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.StatelessIf' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatelessIfOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatelessIfOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatelessIfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatelessIfOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatelessIfOp::getArgs() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatelessIfOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatelessIfOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessIfOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatelessIfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatelessIfOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatelessIfOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr StatelessIfOp::getThenBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getThenBranchAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessIfOp::getThenBranch() {
  auto attr = getThenBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessIfOp::getElseBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getElseBranchAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessIfOp::getElseBranch() {
  auto attr = getElseBranchAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::TypeAttr StatelessIfOp::getTcondAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getTcondAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> StatelessIfOp::getTcond() {
  auto attr = getTcondAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::ArrayAttr StatelessIfOp::getTinAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getTinAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessIfOp::getTin() {
  auto attr = getTinAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessIfOp::getToutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getToutAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessIfOp::getTout() {
  auto attr = getToutAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessIfOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessIfOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatelessIfOp::setThenBranchAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getThenBranchAttrName(), attr);
}

void StatelessIfOp::setElseBranchAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getElseBranchAttrName(), attr);
}

void StatelessIfOp::setTcondAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTcondAttrName(), attr);
}

void StatelessIfOp::setTinAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTinAttrName(), attr);
}

void StatelessIfOp::setToutAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getToutAttrName(), attr);
}

void StatelessIfOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute StatelessIfOp::removeTcondAttr() {
  return (*this)->removeAttr(getTcondAttrName());
}

::mlir::Attribute StatelessIfOp::removeTinAttr() {
  return (*this)->removeAttr(getTinAttrName());
}

::mlir::Attribute StatelessIfOp::removeToutAttr() {
  return (*this)->removeAttr(getToutAttrName());
}

::mlir::Attribute StatelessIfOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void StatelessIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch) {
      build(odsBuilder, odsState, results, cond, args, then_branch, else_branch,
            /*Tcond=*/{}, /*Tin=*/{}, /*Tout=*/{}, /*output_shapes=*/{});
    
}

void StatelessIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addAttribute(getThenBranchAttrName(odsState.name), then_branch);
  odsState.addAttribute(getElseBranchAttrName(odsState.name), else_branch);
  if (Tcond) {
    odsState.addAttribute(getTcondAttrName(odsState.name), Tcond);
  }
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessIfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(cond);
  odsState.addOperands(args);
  odsState.addAttribute(getThenBranchAttrName(odsState.name), then_branch);
  odsState.addAttribute(getElseBranchAttrName(odsState.name), else_branch);
  if (Tcond) {
    odsState.addAttribute(getTcondAttrName(odsState.name), Tcond);
  }
  if (Tin) {
    odsState.addAttribute(getTinAttrName(odsState.name), Tin);
  }
  if (Tout) {
    odsState.addAttribute(getToutAttrName(odsState.name), Tout);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessIfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatelessIfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_branch;
  ::mlir::Attribute tblgen_Tcond;
  ::mlir::Attribute tblgen_Tin;
  ::mlir::Attribute tblgen_Tout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'else_branch'");
    if (namedAttrIt->getName() == getElseBranchAttrName()) {
      tblgen_else_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTcondAttrName()) {
      tblgen_Tcond = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTinAttrName()) {
      tblgen_Tin = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getToutAttrName()) {
      tblgen_Tout = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_then_branch;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'then_branch'");
    if (namedAttrIt->getName() == getThenBranchAttrName()) {
      tblgen_then_branch = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_then_branch, "then_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_else_branch, "else_branch")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops17(*this, tblgen_Tcond, "Tcond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tin, "Tin")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_Tout, "Tout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessIfOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatelessIfOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyIfLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessIfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessIfRegionOp definitions
//===----------------------------------------------------------------------===//

StatelessIfRegionOpAdaptor::StatelessIfRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatelessIfRegion", odsAttrs.getContext());
}

StatelessIfRegionOpAdaptor::StatelessIfRegionOpAdaptor(StatelessIfRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatelessIfRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatelessIfRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatelessIfRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatelessIfRegionOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange StatelessIfRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatelessIfRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DictionaryAttr StatelessIfRegionOpAdaptor::getThenAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessIfRegionOp::getThenAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessIfRegionOpAdaptor::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatelessIfRegionOpAdaptor::getElseAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessIfRegionOp::getElseAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessIfRegionOpAdaptor::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessIfRegionOpAdaptor::getThenRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessIfRegionOp::getThenRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessIfRegionOpAdaptor::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessIfRegionOpAdaptor::getElseRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatelessIfRegionOp::getElseRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessIfRegionOpAdaptor::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange StatelessIfRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &StatelessIfRegionOpAdaptor::getThenRegion() {
  return *odsRegions[0];
}

::mlir::Region &StatelessIfRegionOpAdaptor::getElseRegion() {
  return *odsRegions[1];
}

::mlir::LogicalResult StatelessIfRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_attrs;
  ::mlir::Attribute tblgen_else_region_attrs;
  ::mlir::Attribute tblgen_then_attrs;
  ::mlir::Attribute tblgen_then_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StatelessIfRegionOp::getElseAttrsAttrName(*odsOpName)) {
      tblgen_else_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessIfRegionOp::getElseRegionAttrsAttrName(*odsOpName)) {
      tblgen_else_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessIfRegionOp::getThenAttrsAttrName(*odsOpName)) {
      tblgen_then_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessIfRegionOp::getThenRegionAttrsAttrName(*odsOpName)) {
      tblgen_then_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_then_attrs && !((tblgen_then_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'then_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_else_attrs && !((tblgen_else_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'else_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_then_region_attrs && !((tblgen_then_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'then_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_else_region_attrs && !((tblgen_else_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatelessIfRegion' op ""attribute 'else_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatelessIfRegionOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatelessIfRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> StatelessIfRegionOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range StatelessIfRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatelessIfRegionOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StatelessIfRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessIfRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatelessIfRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatelessIfRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatelessIfRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &StatelessIfRegionOp::getThenRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &StatelessIfRegionOp::getElseRegion() {
  return (*this)->getRegion(1);
}

::mlir::DictionaryAttr StatelessIfRegionOp::getThenAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getThenAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessIfRegionOp::getThenAttrs() {
  auto attr = getThenAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatelessIfRegionOp::getElseAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElseAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessIfRegionOp::getElseAttrs() {
  auto attr = getElseAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessIfRegionOp::getThenRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getThenRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessIfRegionOp::getThenRegionAttrs() {
  auto attr = getThenRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessIfRegionOp::getElseRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElseRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessIfRegionOp::getElseRegionAttrs() {
  auto attr = getElseRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void StatelessIfRegionOp::setThenAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getThenAttrsAttrName(), attr);
}

void StatelessIfRegionOp::setElseAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getElseAttrsAttrName(), attr);
}

void StatelessIfRegionOp::setThenRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getThenRegionAttrsAttrName(), attr);
}

void StatelessIfRegionOp::setElseRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getElseRegionAttrsAttrName(), attr);
}

::mlir::Attribute StatelessIfRegionOp::removeThen_attrsAttr() {
  return (*this)->removeAttr(getThenAttrsAttrName());
}

::mlir::Attribute StatelessIfRegionOp::removeElse_attrsAttr() {
  return (*this)->removeAttr(getElseAttrsAttrName());
}

::mlir::Attribute StatelessIfRegionOp::removeThen_region_attrsAttr() {
  return (*this)->removeAttr(getThenRegionAttrsAttrName());
}

::mlir::Attribute StatelessIfRegionOp::removeElse_region_attrsAttr() {
  return (*this)->removeAttr(getElseRegionAttrsAttrName());
}

void StatelessIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.addAttribute(getThenAttrsAttrName(odsState.name), then_attrs);
  }
  if (else_attrs) {
    odsState.addAttribute(getElseAttrsAttrName(odsState.name), else_attrs);
  }
  if (then_region_attrs) {
    odsState.addAttribute(getThenRegionAttrsAttrName(odsState.name), then_region_attrs);
  }
  if (else_region_attrs) {
    odsState.addAttribute(getElseRegionAttrsAttrName(odsState.name), else_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessIfRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs) {
  odsState.addOperands(cond);
  odsState.addOperands(ctls);
  if (then_attrs) {
    odsState.addAttribute(getThenAttrsAttrName(odsState.name), then_attrs);
  }
  if (else_attrs) {
    odsState.addAttribute(getElseAttrsAttrName(odsState.name), else_attrs);
  }
  if (then_region_attrs) {
    odsState.addAttribute(getThenRegionAttrsAttrName(odsState.name), then_region_attrs);
  }
  if (else_region_attrs) {
    odsState.addAttribute(getElseRegionAttrsAttrName(odsState.name), else_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessIfRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatelessIfRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_else_attrs;
  ::mlir::Attribute tblgen_else_region_attrs;
  ::mlir::Attribute tblgen_then_attrs;
  ::mlir::Attribute tblgen_then_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getElseAttrsAttrName()) {
      tblgen_else_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getElseRegionAttrsAttrName()) {
      tblgen_else_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThenAttrsAttrName()) {
      tblgen_then_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThenRegionAttrsAttrName()) {
      tblgen_then_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_then_attrs, "then_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_else_attrs, "else_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_then_region_attrs, "then_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_else_region_attrs, "else_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "then_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "else_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessIfRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatelessIfRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand condRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> condOperands(condRawOperands);  ::llvm::SMLoc condOperandsLoc;
  (void)condOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> then_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> else_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> condTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;

  condOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(condRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("then"))
    return ::mlir::failure();

  if (parser.parseRegion(*then_regionRegion))
    return ::mlir::failure();

  if (then_regionRegion->empty()) then_regionRegion->emplaceBlock();
  if (parser.parseKeyword("else"))
    return ::mlir::failure();

  if (parser.parseRegion(*else_regionRegion))
    return ::mlir::failure();

  if (else_regionRegion->empty()) else_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType cond__outs_functionType;
  if (parser.parseType(cond__outs_functionType))
    return ::mlir::failure();
  condTypes = cond__outs_functionType.getInputs();
  outsTypes = cond__outs_functionType.getResults();
  result.addRegion(std::move(then_regionRegion));
  result.addRegion(std::move(else_regionRegion));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(condOperands, condTypes, condOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessIfRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCond();
  if (!getCtls().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ' << "then";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getThenRegion());
  _odsPrinter << ' ' << "else";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getElseRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getCond().getType()), getOuts().getTypes());
}

StringRef StatelessIfRegionOp::getDefaultDialect() { return "tfg"; }
void StatelessIfRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

YieldOp StatelessIfRegionOp::getThenYield() {
  return cast<YieldOp>(getThenBlock().getTerminator());
}
YieldOp StatelessIfRegionOp::getElseYield() {
  return cast<YieldOp>(getElseBlock().getTerminator());
}

void StatelessIfRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  GetIfLikeRegionOpSuccessorRegions(*this, index, operands, regions);
}
void StatelessIfRegionOp::getRegionInvocationBounds(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<InvocationBounds> &invocationBounds) {
  invocationBounds.append(/*NumInputs=*/2, /*Elt=*/{0, 1});
}

LogicalResult StatelessIfRegionOp::verify() { return VerifyIfLikeRegionOp(*this); }

BlockArgument StatelessIfRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatelessIfRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatelessIfRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatelessIfRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessIfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessWhileOp definitions
//===----------------------------------------------------------------------===//

StatelessWhileOpAdaptor::StatelessWhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatelessWhile", odsAttrs.getContext());
}

StatelessWhileOpAdaptor::StatelessWhileOpAdaptor(StatelessWhileOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatelessWhileOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatelessWhileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange StatelessWhileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange StatelessWhileOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr StatelessWhileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr StatelessWhileOpAdaptor::getCondAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, StatelessWhileOp::getCondAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatelessWhileOpAdaptor::getCond() {
  auto attr = getCondAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessWhileOpAdaptor::getBodyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessWhileOp::getBodyAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr StatelessWhileOpAdaptor::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::IntegerAttr StatelessWhileOpAdaptor::getParallelIterationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, StatelessWhileOp::getParallelIterationsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t StatelessWhileOpAdaptor::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr StatelessWhileOpAdaptor::getTAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, StatelessWhileOp::getTAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessWhileOpAdaptor::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessWhileOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, StatelessWhileOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessWhileOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult StatelessWhileOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessWhile' op ""requires attribute 'body'");
    if (namedAttrIt->getName() == StatelessWhileOp::getBodyAttrName(*odsOpName)) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatelessWhileOp::getTAttrName(*odsOpName)) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_cond;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessWhile' op ""requires attribute 'cond'");
    if (namedAttrIt->getName() == StatelessWhileOp::getCondAttrName(*odsOpName)) {
      tblgen_cond = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessWhile' op ""requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == StatelessWhileOp::getParallelIterationsAttrName(*odsOpName)) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatelessWhileOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_cond && !((tblgen_cond.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'cond' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_body && !((tblgen_body.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_parallel_iterations && !(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_T && !(((tblgen_T.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_T.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'T' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.StatelessWhile' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void StatelessWhileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> StatelessWhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range StatelessWhileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range StatelessWhileOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange StatelessWhileOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessWhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatelessWhileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatelessWhileOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatelessWhileOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr StatelessWhileOp::getCondAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getCondAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessWhileOp::getCond() {
  auto attr = getCondAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessWhileOp::getBodyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr StatelessWhileOp::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::IntegerAttr StatelessWhileOp::getParallelIterationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getParallelIterationsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t StatelessWhileOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr StatelessWhileOp::getTAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getTAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessWhileOp::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StatelessWhileOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StatelessWhileOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void StatelessWhileOp::setCondAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getCondAttrName(), attr);
}

void StatelessWhileOp::setBodyAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getBodyAttrName(), attr);
}

void StatelessWhileOp::setParallelIterationsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getParallelIterationsAttrName(), attr);
}

void StatelessWhileOp::setTAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTAttrName(), attr);
}

void StatelessWhileOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute StatelessWhileOp::removeTAttr() {
  return (*this)->removeAttr(getTAttrName());
}

::mlir::Attribute StatelessWhileOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations) {
      build(odsBuilder, odsState, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatelessWhileOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'body'");
    if (namedAttrIt->getName() == getBodyAttrName()) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTAttrName()) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_cond;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cond'");
    if (namedAttrIt->getName() == getCondAttrName()) {
      tblgen_cond = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == getParallelIterationsAttrName()) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_cond, "cond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessWhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult StatelessWhileOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyWhileLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessWhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessWhileRegionOp definitions
//===----------------------------------------------------------------------===//

StatelessWhileRegionOpAdaptor::StatelessWhileRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.StatelessWhileRegion", odsAttrs.getContext());
}

StatelessWhileRegionOpAdaptor::StatelessWhileRegionOpAdaptor(StatelessWhileRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatelessWhileRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatelessWhileRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatelessWhileRegionOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange StatelessWhileRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange StatelessWhileRegionOpAdaptor::getInit() {
  return getODSOperands(0);
}

::mlir::ValueRange StatelessWhileRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr StatelessWhileRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr StatelessWhileRegionOpAdaptor::getParallelIterationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, StatelessWhileRegionOp::getParallelIterationsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t StatelessWhileRegionOpAdaptor::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr StatelessWhileRegionOpAdaptor::getCondAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessWhileRegionOp::getCondAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessWhileRegionOpAdaptor::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatelessWhileRegionOpAdaptor::getBodyAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessWhileRegionOp::getBodyAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessWhileRegionOpAdaptor::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessWhileRegionOpAdaptor::getCondRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessWhileRegionOp::getCondRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOpAdaptor::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessWhileRegionOpAdaptor::getBodyRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, StatelessWhileRegionOp::getBodyRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOpAdaptor::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange StatelessWhileRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &StatelessWhileRegionOpAdaptor::getCondRegion() {
  return *odsRegions[0];
}

::mlir::Region &StatelessWhileRegionOpAdaptor::getBodyRegion() {
  return *odsRegions[1];
}

::mlir::LogicalResult StatelessWhileRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  ::mlir::Attribute tblgen_body_region_attrs;
  ::mlir::Attribute tblgen_cond_attrs;
  ::mlir::Attribute tblgen_cond_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessWhileRegion' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == StatelessWhileRegionOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatelessWhileRegionOp::getBodyAttrsAttrName(*odsOpName)) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessWhileRegionOp::getBodyRegionAttrsAttrName(*odsOpName)) {
      tblgen_body_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessWhileRegionOp::getCondAttrsAttrName(*odsOpName)) {
      tblgen_cond_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatelessWhileRegionOp::getCondRegionAttrsAttrName(*odsOpName)) {
      tblgen_cond_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.StatelessWhileRegion' op ""requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == StatelessWhileRegionOp::getParallelIterationsAttrName(*odsOpName)) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'tfg.StatelessWhileRegion' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_parallel_iterations && !(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_cond_attrs && !((tblgen_cond_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'cond_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_body_attrs && !((tblgen_body_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_cond_region_attrs && !((tblgen_cond_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'cond_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_body_region_attrs && !((tblgen_body_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.StatelessWhileRegion' op ""attribute 'body_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatelessWhileRegionOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range StatelessWhileRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range StatelessWhileRegionOp::getInit() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range StatelessWhileRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange StatelessWhileRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange StatelessWhileRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> StatelessWhileRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range StatelessWhileRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range StatelessWhileRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> StatelessWhileRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &StatelessWhileRegionOp::getCondRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &StatelessWhileRegionOp::getBodyRegion() {
  return (*this)->getRegion(1);
}

::mlir::IntegerAttr StatelessWhileRegionOp::getParallelIterationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getParallelIterationsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t StatelessWhileRegionOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr StatelessWhileRegionOp::getCondAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCondAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessWhileRegionOp::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr StatelessWhileRegionOp::getBodyAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > StatelessWhileRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessWhileRegionOp::getCondRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCondRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOp::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr StatelessWhileRegionOp::getBodyRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> StatelessWhileRegionOp::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void StatelessWhileRegionOp::setParallelIterationsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getParallelIterationsAttrName(), attr);
}

void StatelessWhileRegionOp::setCondAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getCondAttrsAttrName(), attr);
}

void StatelessWhileRegionOp::setBodyAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getBodyAttrsAttrName(), attr);
}

void StatelessWhileRegionOp::setCondRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getCondRegionAttrsAttrName(), attr);
}

void StatelessWhileRegionOp::setBodyRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getBodyRegionAttrsAttrName(), attr);
}

::mlir::Attribute StatelessWhileRegionOp::removeCond_attrsAttr() {
  return (*this)->removeAttr(getCondAttrsAttrName());
}

::mlir::Attribute StatelessWhileRegionOp::removeBody_attrsAttr() {
  return (*this)->removeAttr(getBodyAttrsAttrName());
}

::mlir::Attribute StatelessWhileRegionOp::removeCond_region_attrsAttr() {
  return (*this)->removeAttr(getCondRegionAttrsAttrName());
}

::mlir::Attribute StatelessWhileRegionOp::removeBody_region_attrsAttr() {
  return (*this)->removeAttr(getBodyRegionAttrsAttrName());
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatelessWhileRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StatelessWhileRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  ::mlir::Attribute tblgen_body_region_attrs;
  ::mlir::Attribute tblgen_cond_attrs;
  ::mlir::Attribute tblgen_cond_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBodyAttrsAttrName()) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getBodyRegionAttrsAttrName()) {
      tblgen_body_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCondAttrsAttrName()) {
      tblgen_cond_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCondRegionAttrsAttrName()) {
      tblgen_cond_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == getParallelIterationsAttrName()) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_cond_attrs, "cond_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_cond_region_attrs, "cond_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_body_region_attrs, "body_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "cond_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatelessWhileRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult StatelessWhileRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> cond_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> initTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*cond_regionRegion))
    return ::mlir::failure();

  if (cond_regionRegion->empty()) cond_regionRegion->emplaceBlock();
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  ::mlir::FunctionType init__outs_functionType;
  if (parser.parseType(init__outs_functionType))
    return ::mlir::failure();
  initTypes = init__outs_functionType.getInputs();
  outsTypes = init__outs_functionType.getResults();
  }
  result.addRegion(std::move(cond_regionRegion));
  result.addRegion(std::move(body_regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StatelessWhileRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getCondRegion());
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  if (!getInit().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printFunctionalType(getInit().getTypes(), getOuts().getTypes());
  }
}

StringRef StatelessWhileRegionOp::getDefaultDialect() { return "tfg"; }
void StatelessWhileRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

ConditionOp StatelessWhileRegionOp::getCondCondition() {
  return cast<ConditionOp>(getCondBlock().getTerminator());
}
YieldOp StatelessWhileRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}

OperandRange StatelessWhileRegionOp::getSuccessorEntryOperands(
    Optional<unsigned> index) {
  return getInit();
}
void StatelessWhileRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return GetWhileLikeRegionOpSuccessorRegions(*this, index, operands,
                                              regions);
}

LogicalResult StatelessWhileRegionOp::verify() { return VerifyWhileLikeRegionOp(*this); }

BlockArgument StatelessWhileRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument StatelessWhileRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument StatelessWhileRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument StatelessWhileRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessWhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::WhileOp definitions
//===----------------------------------------------------------------------===//

WhileOpAdaptor::WhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.While", odsAttrs.getContext());
}

WhileOpAdaptor::WhileOpAdaptor(WhileOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WhileOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WhileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange WhileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WhileOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WhileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::tf_type::FuncAttr WhileOpAdaptor::getCondAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, WhileOp::getCondAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr WhileOpAdaptor::getCond() {
  auto attr = getCondAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr WhileOpAdaptor::getBodyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, WhileOp::getBodyAttrName(*odsOpName)).cast<::mlir::tf_type::FuncAttr>();
  return attr;
}

::mlir::tf_type::FuncAttr WhileOpAdaptor::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::IntegerAttr WhileOpAdaptor::getParallelIterationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, WhileOp::getParallelIterationsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t WhileOpAdaptor::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr WhileOpAdaptor::getTAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, WhileOp::getTAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > WhileOpAdaptor::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr WhileOpAdaptor::getOutputShapesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, WhileOp::getOutputShapesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > WhileOpAdaptor::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult WhileOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.While' op ""requires attribute 'body'");
    if (namedAttrIt->getName() == WhileOp::getBodyAttrName(*odsOpName)) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WhileOp::getTAttrName(*odsOpName)) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_cond;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.While' op ""requires attribute 'cond'");
    if (namedAttrIt->getName() == WhileOp::getCondAttrName(*odsOpName)) {
      tblgen_cond = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.While' op ""requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == WhileOp::getParallelIterationsAttrName(*odsOpName)) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WhileOp::getOutputShapesAttrName(*odsOpName)) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_cond && !((tblgen_cond.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.While' op ""attribute 'cond' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_body && !((tblgen_body.isa<::mlir::tf_type::FuncAttr>())))
    return emitError(loc, "'tfg.While' op ""attribute 'body' failed to satisfy constraint: Models the `AttrValue.value.func` proto attribute value as a pair of SymbolRef and DictionaryAttr");

  if (tblgen_parallel_iterations && !(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.While' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_T && !(((tblgen_T.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_T.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'tfg.While' op ""attribute 'T' failed to satisfy constraint: type array attribute");

  if (tblgen_output_shapes && !(((tblgen_output_shapes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_shapes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::tf_type::ShapeAttr>())); }))))
    return emitError(loc, "'tfg.While' op ""attribute 'output_shapes' failed to satisfy constraint: An array of shapes.");
  return ::mlir::success();
}

void WhileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!resultGroup0.empty())
    setNameFn(*resultGroup0.begin(), "outs");
  auto resultGroup1 = getODSResults(1);
  if (!resultGroup1.empty())
    setNameFn(*resultGroup1.begin(), "ctl");
}

std::pair<unsigned, unsigned> WhileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range WhileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WhileOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WhileOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WhileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range WhileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range WhileOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> WhileOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::tf_type::FuncAttr WhileOp::getCondAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getCondAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr WhileOp::getCond() {
  auto attr = getCondAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr WhileOp::getBodyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyAttrName()).cast<::mlir::tf_type::FuncAttr>();
}

::mlir::tf_type::FuncAttr WhileOp::getBody() {
  auto attr = getBodyAttr();
  return attr.cast<::mlir::tf_type::FuncAttr>();
}

::mlir::IntegerAttr WhileOp::getParallelIterationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getParallelIterationsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t WhileOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::ArrayAttr WhileOp::getTAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getTAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > WhileOp::getT() {
  auto attr = getTAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr WhileOp::getOutputShapesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getOutputShapesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > WhileOp::getOutputShapes() {
  auto attr = getOutputShapesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void WhileOp::setCondAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getCondAttrName(), attr);
}

void WhileOp::setBodyAttr(::mlir::tf_type::FuncAttr attr) {
  (*this)->setAttr(getBodyAttrName(), attr);
}

void WhileOp::setParallelIterationsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getParallelIterationsAttrName(), attr);
}

void WhileOp::setTAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTAttrName(), attr);
}

void WhileOp::setOutputShapesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOutputShapesAttrName(), attr);
}

::mlir::Attribute WhileOp::removeTAttr() {
  return (*this)->removeAttr(getTAttrName());
}

::mlir::Attribute WhileOp::removeOutput_shapesAttr() {
  return (*this)->removeAttr(getOutputShapesAttrName());
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations) {
      build(odsBuilder, odsState, results, args, cond, body, parallel_iterations,
            /*T=*/{}, /*output_shapes=*/{});
    
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes) {
  odsState.addOperands(args);
  odsState.addAttribute(getCondAttrName(odsState.name), cond);
  odsState.addAttribute(getBodyAttrName(odsState.name), body);
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (T) {
    odsState.addAttribute(getTAttrName(odsState.name), T);
  }
  if (output_shapes) {
    odsState.addAttribute(getOutputShapesAttrName(odsState.name), output_shapes);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WhileOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_body;
  ::mlir::Attribute tblgen_T;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'body'");
    if (namedAttrIt->getName() == getBodyAttrName()) {
      tblgen_body = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getTAttrName()) {
      tblgen_T = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_cond;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cond'");
    if (namedAttrIt->getName() == getCondAttrName()) {
      tblgen_cond = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  ::mlir::Attribute tblgen_output_shapes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == getParallelIterationsAttrName()) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getOutputShapesAttrName()) {
      tblgen_output_shapes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_cond, "cond")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops5(*this, tblgen_body, "body")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops1(*this, tblgen_T, "T")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops2(*this, tblgen_output_shapes, "output_shapes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WhileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

LogicalResult WhileOp::verifySymbolUses(
    SymbolTableCollection &symbol_table) {
  return VerifyWhileLikeOp(*this, symbol_table);
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::WhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::WhileRegionOp definitions
//===----------------------------------------------------------------------===//

WhileRegionOpAdaptor::WhileRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.WhileRegion", odsAttrs.getContext());
}

WhileRegionOpAdaptor::WhileRegionOpAdaptor(WhileRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WhileRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WhileRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WhileRegionOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange WhileRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WhileRegionOpAdaptor::getInit() {
  return getODSOperands(0);
}

::mlir::ValueRange WhileRegionOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr WhileRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr WhileRegionOpAdaptor::getParallelIterationsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WhileRegionOp::getParallelIterationsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t WhileRegionOpAdaptor::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr WhileRegionOpAdaptor::getCondAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, WhileRegionOp::getCondAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > WhileRegionOpAdaptor::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr WhileRegionOpAdaptor::getBodyAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, WhileRegionOp::getBodyAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > WhileRegionOpAdaptor::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr WhileRegionOpAdaptor::getCondRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, WhileRegionOp::getCondRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> WhileRegionOpAdaptor::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr WhileRegionOpAdaptor::getBodyRegionAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, WhileRegionOp::getBodyRegionAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
  return attr;
}

::llvm::Optional<::mlir::tfg::RegionAttr> WhileRegionOpAdaptor::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::RegionRange WhileRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &WhileRegionOpAdaptor::getCondRegion() {
  return *odsRegions[0];
}

::mlir::Region &WhileRegionOpAdaptor::getBodyRegion() {
  return *odsRegions[1];
}

::mlir::LogicalResult WhileRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  ::mlir::Attribute tblgen_body_region_attrs;
  ::mlir::Attribute tblgen_cond_attrs;
  ::mlir::Attribute tblgen_cond_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.WhileRegion' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == WhileRegionOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WhileRegionOp::getBodyAttrsAttrName(*odsOpName)) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WhileRegionOp::getBodyRegionAttrsAttrName(*odsOpName)) {
      tblgen_body_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WhileRegionOp::getCondAttrsAttrName(*odsOpName)) {
      tblgen_cond_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WhileRegionOp::getCondRegionAttrsAttrName(*odsOpName)) {
      tblgen_cond_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.WhileRegion' op ""requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == WhileRegionOp::getParallelIterationsAttrName(*odsOpName)) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'tfg.WhileRegion' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_parallel_iterations && !(((tblgen_parallel_iterations.isa<::mlir::IntegerAttr>())) && ((tblgen_parallel_iterations.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'parallel_iterations' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_cond_attrs && !((tblgen_cond_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'cond_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_body_attrs && !((tblgen_body_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'body_attrs' failed to satisfy constraint: dictionary of named attribute values");

  if (tblgen_cond_region_attrs && !((tblgen_cond_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'cond_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");

  if (tblgen_body_region_attrs && !((tblgen_body_region_attrs.isa<::mlir::tfg::RegionAttr>())))
    return emitError(loc, "'tfg.WhileRegion' op ""attribute 'body_region_attrs' failed to satisfy constraint: Region attributes, argument and result attributes.");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WhileRegionOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range WhileRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WhileRegionOp::getInit() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range WhileRegionOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange WhileRegionOp::getInitMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WhileRegionOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> WhileRegionOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range WhileRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range WhileRegionOp::getOuts() {
  return getODSResults(0);
}

::mlir::TypedValue<::mlir::tf_type::ControlType> WhileRegionOp::getCtl() {
  return *getODSResults(1).begin();
}

::mlir::Region &WhileRegionOp::getCondRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &WhileRegionOp::getBodyRegion() {
  return (*this)->getRegion(1);
}

::mlir::IntegerAttr WhileRegionOp::getParallelIterationsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getParallelIterationsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t WhileRegionOp::getParallelIterations() {
  auto attr = getParallelIterationsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::DictionaryAttr WhileRegionOp::getCondAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCondAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > WhileRegionOp::getCondAttrs() {
  auto attr = getCondAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::DictionaryAttr WhileRegionOp::getBodyAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > WhileRegionOp::getBodyAttrs() {
  auto attr = getBodyAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::tfg::RegionAttr WhileRegionOp::getCondRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getCondRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> WhileRegionOp::getCondRegionAttrs() {
  auto attr = getCondRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

::mlir::tfg::RegionAttr WhileRegionOp::getBodyRegionAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBodyRegionAttrsAttrName()).dyn_cast_or_null<::mlir::tfg::RegionAttr>();
}

::llvm::Optional<::mlir::tfg::RegionAttr> WhileRegionOp::getBodyRegionAttrs() {
  auto attr = getBodyRegionAttrsAttr();
  return attr ? ::llvm::Optional<::mlir::tfg::RegionAttr>(attr.cast<::mlir::tfg::RegionAttr>()) : (::llvm::None);
}

void WhileRegionOp::setParallelIterationsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getParallelIterationsAttrName(), attr);
}

void WhileRegionOp::setCondAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getCondAttrsAttrName(), attr);
}

void WhileRegionOp::setBodyAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getBodyAttrsAttrName(), attr);
}

void WhileRegionOp::setCondRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getCondRegionAttrsAttrName(), attr);
}

void WhileRegionOp::setBodyRegionAttrsAttr(::mlir::tfg::RegionAttr attr) {
  (*this)->setAttr(getBodyRegionAttrsAttrName(), attr);
}

::mlir::Attribute WhileRegionOp::removeCond_attrsAttr() {
  return (*this)->removeAttr(getCondAttrsAttrName());
}

::mlir::Attribute WhileRegionOp::removeBody_attrsAttr() {
  return (*this)->removeAttr(getBodyAttrsAttrName());
}

::mlir::Attribute WhileRegionOp::removeCond_region_attrsAttr() {
  return (*this)->removeAttr(getCondRegionAttrsAttrName());
}

::mlir::Attribute WhileRegionOp::removeBody_region_attrsAttr() {
  return (*this)->removeAttr(getBodyRegionAttrsAttrName());
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), parallel_iterations);
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  odsState.addTypes(outs);
  odsState.addTypes(ctl);
}

void WhileRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs) {
  odsState.addOperands(init);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(init.size()), static_cast<int32_t>(ctls.size())}));
  odsState.addAttribute(getParallelIterationsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), parallel_iterations));
  if (cond_attrs) {
    odsState.addAttribute(getCondAttrsAttrName(odsState.name), cond_attrs);
  }
  if (body_attrs) {
    odsState.addAttribute(getBodyAttrsAttrName(odsState.name), body_attrs);
  }
  if (cond_region_attrs) {
    odsState.addAttribute(getCondRegionAttrsAttrName(odsState.name), cond_region_attrs);
  }
  if (body_region_attrs) {
    odsState.addAttribute(getBodyRegionAttrsAttrName(odsState.name), body_region_attrs);
  }
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WhileRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 2; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WhileRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_body_attrs;
  ::mlir::Attribute tblgen_body_region_attrs;
  ::mlir::Attribute tblgen_cond_attrs;
  ::mlir::Attribute tblgen_cond_region_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBodyAttrsAttrName()) {
      tblgen_body_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getBodyRegionAttrsAttrName()) {
      tblgen_body_region_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCondAttrsAttrName()) {
      tblgen_cond_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCondRegionAttrsAttrName()) {
      tblgen_cond_region_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_parallel_iterations;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'parallel_iterations'");
    if (namedAttrIt->getName() == getParallelIterationsAttrName()) {
      tblgen_parallel_iterations = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops18(*this, tblgen_parallel_iterations, "parallel_iterations")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_cond_attrs, "cond_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops6(*this, tblgen_body_attrs, "body_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_cond_region_attrs, "cond_region_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_ops7(*this, tblgen_body_region_attrs, "body_region_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "cond_region", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_ops0(*this, region, "body_region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult WhileRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WhileRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> initOperands;
  ::llvm::SMLoc initOperandsLoc;
  (void)initOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  std::unique_ptr<::mlir::Region> cond_regionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> body_regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::ArrayRef<::mlir::Type> initTypes;
  ::llvm::ArrayRef<::mlir::Type> outsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  initOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(initOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*cond_regionRegion))
    return ::mlir::failure();

  if (cond_regionRegion->empty()) cond_regionRegion->emplaceBlock();
  if (parser.parseKeyword("do"))
    return ::mlir::failure();

  if (parser.parseRegion(*body_regionRegion))
    return ::mlir::failure();

  if (body_regionRegion->empty()) body_regionRegion->emplaceBlock();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  ::mlir::FunctionType init__outs_functionType;
  if (parser.parseType(init__outs_functionType))
    return ::mlir::failure();
  initTypes = init__outs_functionType.getInputs();
  outsTypes = init__outs_functionType.getResults();
  }
  result.addRegion(std::move(cond_regionRegion));
  result.addRegion(std::move(body_regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(initOperands.size()), static_cast<int32_t>(ctlsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  result.addTypes(outsTypes);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(initOperands, initTypes, initOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WhileRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInit().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInit();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getCondRegion());
  _odsPrinter << ' ' << "do";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBodyRegion());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  if (!getInit().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter.printFunctionalType(getInit().getTypes(), getOuts().getTypes());
  }
}

StringRef WhileRegionOp::getDefaultDialect() { return "tfg"; }
void WhileRegionOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  GenericGetAsmResultNames(*this, setNameFn);
}

ConditionOp WhileRegionOp::getCondCondition() {
  return cast<ConditionOp>(getCondBlock().getTerminator());
}
YieldOp WhileRegionOp::getBodyYield() {
  return cast<YieldOp>(getBodyBlock().getTerminator());
}

OperandRange WhileRegionOp::getSuccessorEntryOperands(
    Optional<unsigned> index) {
  return getInit();
}
void WhileRegionOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  return GetWhileLikeRegionOpSuccessorRegions(*this, index, operands,
                                              regions);
}

LogicalResult WhileRegionOp::verify() { return VerifyWhileLikeRegionOp(*this); }

BlockArgument WhileRegionOp::getDataValueOf(BlockArgument ctl) {
  return GetLoopRegionDataOf(ctl);
}
BlockArgument WhileRegionOp::getControlTokenOf(BlockArgument data) {
  return GetLoopRegionControlOf(data);
}
BlockArgument WhileRegionOp::getDataValue(Region &region, unsigned idx) {
  return GetLoopRegionDataArgs(region)[idx];
}
BlockArgument WhileRegionOp::getControlToken(Region &region, unsigned idx) {
  return GetLoopRegionControlTokens(region)[idx];
}
} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::WhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("tfg.yield", odsAttrs.getContext());
}

YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, YieldOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::getArgs() {
  return getODSOperands(0);
}

::mlir::ValueRange YieldOpAdaptor::getCtls() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'tfg.yield' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == YieldOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'tfg.yield' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::getArgs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range YieldOp::getCtls() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange YieldOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange YieldOp::getCtlsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}));
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::ValueRange ctls) {
  odsState.addOperands(args);
  odsState.addOperands(ctls);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(args.size()), static_cast<int32_t>(ctls.size())}));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_ops3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> ctlsOperands;
  ::llvm::SMLoc ctlsOperandsLoc;
  (void)ctlsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalLSquare())) {

  ctlsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(ctlsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(argsOperands.size()), static_cast<int32_t>(ctlsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::tf_type::ControlType>();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ctlsOperands, odsBuildableType0, ctlsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getArgs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ")";
  }
  if (!getCtls().empty()) {
    _odsPrinter << ' ' << "[";
    _odsPrinter << getCtls();
    _odsPrinter << "]";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  if (!getArgs().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
  }
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace tfg
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::YieldOp)


#endif  // GET_OP_CLASSES

