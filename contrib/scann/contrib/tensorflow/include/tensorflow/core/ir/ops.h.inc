/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace tfg {
class CaseOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class CaseRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class ConditionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class ForOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class ForRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class GetResultOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class GraphFuncOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class GraphOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class IfOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class IfRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class ReturnOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatefulCaseOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatefulCaseRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatefulIfOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatefulIfRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatefulWhileOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatefulWhileRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatelessCaseOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatelessCaseRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatelessIfOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatelessIfRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatelessWhileOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class StatelessWhileRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class WhileOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class WhileRegionOp;
} // namespace tfg
} // namespace mlir
namespace mlir {
namespace tfg {
class YieldOp;
} // namespace tfg
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::CaseOp declarations
//===----------------------------------------------------------------------===//

class CaseOpAdaptor {
public:
  CaseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CaseOpAdaptor(CaseOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getBranchesAttr();
  ::mlir::ArrayAttr getBranches();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CaseOp : public ::mlir::Op<CaseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CaseOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tin"), ::llvm::StringRef("Tout"), ::llvm::StringRef("branches"), ::llvm::StringRef("output_shapes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTinAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getToutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getToutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBranchesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBranchesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.Case");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getBranchIndexMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::ArrayAttr getBranchesAttr();
  ::mlir::ArrayAttr getBranches();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setBranchesAttr(::mlir::ArrayAttr attr);
  void setTinAttr(::mlir::ArrayAttr attr);
  void setToutAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTinAttr();
  ::mlir::Attribute removeToutAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::CaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::CaseRegionOp declarations
//===----------------------------------------------------------------------===//

class CaseRegionOpAdaptor {
public:
  CaseRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CaseRegionOpAdaptor(CaseRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getBranchAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getBranchAttrs();
  ::mlir::ArrayAttr getRegionAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::RegionRange getBranches();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CaseRegionOp : public ::mlir::Op<CaseRegionOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CaseRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("branch_attrs"), ::llvm::StringRef("region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBranchAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBranchAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.CaseRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getBranchIndex();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getBranchIndexMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::MutableArrayRef<::mlir::Region> getBranches();
  ::mlir::ArrayAttr getBranchAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getBranchAttrs();
  ::mlir::ArrayAttr getRegionAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getRegionAttrs();
  void setBranchAttrsAttr(::mlir::ArrayAttr attr);
  void setRegionAttrsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeBranch_attrsAttr();
  ::mlir::Attribute removeRegion_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getBranchBlock(unsigned idx) { return getBranches()[idx].front(); }
  YieldOp branch_yield(unsigned idx);

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (auto attrs = getRegionAttrsAttr())
      return attrs[index].cast<RegionAttr>();
    return {};
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    SmallVector<Attribute> array = llvm::to_vector(getRegionAttrsAttr());
    array[index] = attrs;
    setRegionAttrsAttr(ArrayAttr::get(getContext(), array));
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::CaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ConditionOp declarations
//===----------------------------------------------------------------------===//

class ConditionOpAdaptor {
public:
  ConditionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConditionOpAdaptor(ConditionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getArgs();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConditionOp : public ::mlir::Op<ConditionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator, ::mlir::RegionBranchTerminatorOpInterface::Trait, ::mlir::OpTrait::IntrinsicOperation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConditionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.condition");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getCond();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::ValueRange ctls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::ValueRange ctls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::llvm::Optional<unsigned> index);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::ConditionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ForOp declarations
//===----------------------------------------------------------------------===//

class ForOpAdaptor {
public:
  ForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ForOpAdaptor(ForOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getStart();
  ::mlir::Value getLimit();
  ::mlir::Value getDelta();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ForOp : public ::mlir::Op<ForOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("body")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.For");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getStart();
  ::mlir::Value getLimit();
  ::mlir::Value getDelta();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getStartMutable();
  ::mlir::MutableOperandRange getLimitMutable();
  ::mlir::MutableOperandRange getDeltaMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  void setBodyAttr(::mlir::tf_type::FuncAttr attr);
  void setTAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value start, Value limit, Value delta, ValueRange args, FuncAttr body);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr body, /*optional*/::mlir::ArrayAttr T);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr body, /*optional*/::mlir::ArrayAttr T);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::ForOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ForRegionOp declarations
//===----------------------------------------------------------------------===//

class ForRegionOpAdaptor {
public:
  ForRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ForRegionOpAdaptor(ForRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getStart();
  ::mlir::Value getLimit();
  ::mlir::Value getDelta();
  ::mlir::ValueRange getInit();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBodyRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ForRegionOp : public ::mlir::Op<ForRegionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("body_attrs"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBodyAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBodyAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getRegionAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.ForRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getStart();
  ::mlir::TypedValue<::mlir::TensorType> getLimit();
  ::mlir::TypedValue<::mlir::TensorType> getDelta();
  ::mlir::Operation::operand_range getInit();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getStartMutable();
  ::mlir::MutableOperandRange getLimitMutable();
  ::mlir::MutableOperandRange getDeltaMutable();
  ::mlir::MutableOperandRange getInitMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getBodyRegion();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getRegionAttrs();
  void setBodyAttrsAttr(::mlir::DictionaryAttr attr);
  void setRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeBody_attrsAttr();
  ::mlir::Attribute removeRegion_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange init, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value start, ::mlir::Value limit, ::mlir::Value delta, ::mlir::ValueRange init, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  ::mlir::OperandRange getSuccessorEntryOperands(::llvm::Optional<unsigned> index);
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getBodyBlock() { return getBodyRegion().front(); }
  YieldOp getBodyYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    return getRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    setRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::ForRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GetResultOp declarations
//===----------------------------------------------------------------------===//

class GetResultOpAdaptor {
public:
  GetResultOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GetResultOpAdaptor(GetResultOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getNameAttr();
  ::llvm::StringRef getName();
  ::mlir::IntegerAttr getNumberAttr();
  uint32_t getNumber();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GetResultOp : public ::mlir::Op<GetResultOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::tf_type::OpaqueTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IntrinsicOperation, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetResultOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name"), ::llvm::StringRef("number")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getNumberAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getNumberAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.get_result");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::tf_type::OpaqueTensorType> getValue();
  ::mlir::MutableOperandRange getValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::tf_type::OpaqueTensorType> getResult();
  ::mlir::StringAttr getNameAttr();
  ::llvm::StringRef getName();
  ::mlir::IntegerAttr getNumberAttr();
  uint32_t getNumber();
  void setNameAttr(::mlir::StringAttr attr);
  void setNumberAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::StringAttr name, ::mlir::IntegerAttr number);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value, ::llvm::StringRef name, uint32_t number);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::llvm::StringRef name, uint32_t number);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::llvm::StringRef name, uint32_t number);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::GetResultOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GraphFuncOp declarations
//===----------------------------------------------------------------------===//

class GraphFuncOpAdaptor {
public:
  GraphFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GraphFuncOpAdaptor(GraphFuncOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr getGenericAttr();
  bool getGeneric();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getFunctionTypeAttr();
  ::mlir::FunctionType getFunctionType();
  ::mlir::ArrayAttr getArgAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getArgAttrs();
  ::mlir::ArrayAttr getResAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getResAttrs();
  ::mlir::StringAttr getDescriptionAttr();
  ::llvm::Optional< ::llvm::StringRef > getDescription();
  ::mlir::UnitAttr getIsStatefulAttr();
  bool getIsStateful();
  ::mlir::FlatSymbolRefAttr getGradientAttr();
  ::llvm::Optional< ::llvm::StringRef > getGradient();
  ::mlir::DenseIntElementsAttr getResourceArgUniqueIdsKeysAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getResourceArgUniqueIdsKeys();
  ::mlir::DenseIntElementsAttr getResourceArgUniqueIdsValuesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getResourceArgUniqueIdsValues();
  ::mlir::ArrayAttr getControlOutputAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getControlOutput();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBody();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GraphFuncOp : public ::mlir::Op<GraphFuncOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::RegionKindInterface::Trait, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::CallableOpInterface::Trait, ::mlir::FunctionOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::SymbolOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait, ::mlir::OpTrait::IntrinsicOperation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GraphFuncOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("arg_attrs"), ::llvm::StringRef("control_output"), ::llvm::StringRef("description"), ::llvm::StringRef("function_type"), ::llvm::StringRef("generic"), ::llvm::StringRef("gradient"), ::llvm::StringRef("is_stateful"), ::llvm::StringRef("res_attrs"), ::llvm::StringRef("resource_arg_unique_ids_keys"), ::llvm::StringRef("resource_arg_unique_ids_values"), ::llvm::StringRef("sym_name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getArgAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getArgAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getControlOutputAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getControlOutputAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getDescriptionAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getDescriptionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getFunctionTypeAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getFunctionTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getGenericAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getGenericAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getGradientAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getGradientAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getIsStatefulAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getIsStatefulAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getResAttrsAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getResAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getResourceArgUniqueIdsKeysAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getResourceArgUniqueIdsKeysAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getResourceArgUniqueIdsValuesAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getResourceArgUniqueIdsValuesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.func");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getBody();
  ::mlir::UnitAttr getGenericAttr();
  bool getGeneric();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getFunctionTypeAttr();
  ::mlir::FunctionType getFunctionType();
  ::mlir::ArrayAttr getArgAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getArgAttrs();
  ::mlir::ArrayAttr getResAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getResAttrs();
  ::mlir::StringAttr getDescriptionAttr();
  ::llvm::Optional< ::llvm::StringRef > getDescription();
  ::mlir::UnitAttr getIsStatefulAttr();
  bool getIsStateful();
  ::mlir::FlatSymbolRefAttr getGradientAttr();
  ::llvm::Optional< ::llvm::StringRef > getGradient();
  ::mlir::DenseIntElementsAttr getResourceArgUniqueIdsKeysAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getResourceArgUniqueIdsKeys();
  ::mlir::DenseIntElementsAttr getResourceArgUniqueIdsValuesAttr();
  ::llvm::Optional< ::mlir::DenseIntElementsAttr > getResourceArgUniqueIdsValues();
  ::mlir::ArrayAttr getControlOutputAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getControlOutput();
  void setGenericAttr(::mlir::UnitAttr attr);
  void setSymNameAttr(::mlir::StringAttr attr);
  void setFunctionTypeAttr(::mlir::TypeAttr attr);
  void setArgAttrsAttr(::mlir::ArrayAttr attr);
  void setResAttrsAttr(::mlir::ArrayAttr attr);
  void setDescriptionAttr(::mlir::StringAttr attr);
  void setIsStatefulAttr(::mlir::UnitAttr attr);
  void setGradientAttr(::mlir::FlatSymbolRefAttr attr);
  void setResourceArgUniqueIdsKeysAttr(::mlir::DenseIntElementsAttr attr);
  void setResourceArgUniqueIdsValuesAttr(::mlir::DenseIntElementsAttr attr);
  void setControlOutputAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeGenericAttr();
  ::mlir::Attribute removeArg_attrsAttr();
  ::mlir::Attribute removeRes_attrsAttr();
  ::mlir::Attribute removeDescriptionAttr();
  ::mlir::Attribute removeIs_statefulAttr();
  ::mlir::Attribute removeGradientAttr();
  ::mlir::Attribute removeResource_arg_unique_ids_keysAttr();
  ::mlir::Attribute removeResource_arg_unique_ids_valuesAttr();
  ::mlir::Attribute removeControl_outputAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, const Twine &sym_name, FunctionType type, bool generic);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr generic, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/::mlir::UnitAttr is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr generic, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/::mlir::UnitAttr is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool generic, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/bool is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool generic, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::ArrayAttr arg_attrs, /*optional*/::mlir::ArrayAttr res_attrs, /*optional*/::mlir::StringAttr description, /*optional*/bool is_stateful, /*optional*/::mlir::FlatSymbolRefAttr gradient, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_keys, /*optional*/::mlir::DenseIntElementsAttr resource_arg_unique_ids_values, /*optional*/::mlir::ArrayAttr control_output);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult canonicalize(GraphFuncOp op, ::mlir::PatternRewriter &rewriter);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  TFGraphDialect *getDialect() {
    return cast<TFGraphDialect>((*this)->getDialect());
  }

  // Returns true if this function is marked for JIT compilation.
  bool isMarkedForCompilation();

  // Return the function called by this operation if this operation is a call
  // and the function exists in the symbol_table, otherwise return null.
  static GraphFuncOp getCalledFunction(Operation *op,
                                       SymbolTable &symbol_table);

  //===------------------------------------------------------------------===//
  // OpAsmOpInterface
  //===------------------------------------------------------------------===//

  // This will filter the `tfg.` prefix in front of operations inside the
  // func body.
  static StringRef getDefaultDialect() {
    return "tfg";
  }

  /// Get a special name to use when printing block arguments.
  void getAsmBlockArgumentNames(Region &region, OpAsmSetValueNameFn set_name_fn);

  //===------------------------------------------------------------------===//
  // CallableOpInterface
  //===------------------------------------------------------------------===//

  // Returns the region on the current operation that is callable. This may
  // return null in the case of an external callable object, e.g. an external
  // function.
  Region *getCallableRegion() {
    return isExternal() ? nullptr : &this->getRegion();
  }

  // Returns the results types that the callable region produces when
  // executed.
  ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

  //===------------------------------------------------------------------===//
  // FunctionOpInterface Methods
  //===------------------------------------------------------------------===//

  /// Returns the argument types of this function.
  ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

  /// Returns the result types of this function.
  ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

  LogicalResult verifyType();
  LogicalResult verifyBody();

  //===------------------------------------------------------------------===//
  // SymbolOpInterface Methods
  //===------------------------------------------------------------------===//

  bool isDeclaration() { return isExternal(); }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::GraphFuncOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::GraphOp declarations
//===----------------------------------------------------------------------===//

class GraphOpAdaptor {
public:
  GraphOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GraphOpAdaptor(GraphOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::VersionAttr getVersionAttr();
  ::mlir::tf_type::VersionAttr getVersion();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getNodes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GraphOp : public ::mlir::Op<GraphOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::NoTerminator, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::OpTrait::IsIsolatedFromAbove, ::mlir::OpAsmOpInterface::Trait, ::mlir::RegionKindInterface::Trait, ::mlir::OpTrait::IntrinsicOperation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GraphOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("version")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getVersionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getVersionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.graph");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getNodes();
  ::mlir::tf_type::VersionAttr getVersionAttr();
  ::mlir::tf_type::VersionAttr getVersion();
  void setVersionAttr(::mlir::tf_type::VersionAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::tf_type::VersionAttr version);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::tf_type::VersionAttr version);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  TFGraphDialect *getDialect() {
    return cast<TFGraphDialect>((*this)->getDialect());
  }

  // Override the getOps() inherited from the OneRegion trait to return a
  // range of TFOp instead of the generic mlir::Operation.
  auto getOps() { return getRegion().template getOps<TFOp>(); }

  //===------------------------------------------------------------------===//
  // OpAsmOpInterface
  //===------------------------------------------------------------------===//

  // This will filter the `tfg.` prefix in front of operations inside the
  // graph body.
  static StringRef getDefaultDialect() {
    return "tfg";
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::GraphOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::IfOp declarations
//===----------------------------------------------------------------------===//

class IfOpAdaptor {
public:
  IfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  IfOpAdaptor(IfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getThenBranchAttr();
  ::mlir::tf_type::FuncAttr getThenBranch();
  ::mlir::tf_type::FuncAttr getElseBranchAttr();
  ::mlir::tf_type::FuncAttr getElseBranch();
  ::mlir::TypeAttr getTcondAttr();
  ::llvm::Optional<::mlir::Type> getTcond();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class IfOp : public ::mlir::Op<IfOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IfOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tcond"), ::llvm::StringRef("Tin"), ::llvm::StringRef("Tout"), ::llvm::StringRef("else_branch"), ::llvm::StringRef("output_shapes"), ::llvm::StringRef("then_branch")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTcondAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTcondAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTinAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getToutAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getToutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getElseBranchAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getElseBranchAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getThenBranchAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getThenBranchAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.If");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getThenBranchAttr();
  ::mlir::tf_type::FuncAttr getThenBranch();
  ::mlir::tf_type::FuncAttr getElseBranchAttr();
  ::mlir::tf_type::FuncAttr getElseBranch();
  ::mlir::TypeAttr getTcondAttr();
  ::llvm::Optional<::mlir::Type> getTcond();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setThenBranchAttr(::mlir::tf_type::FuncAttr attr);
  void setElseBranchAttr(::mlir::tf_type::FuncAttr attr);
  void setTcondAttr(::mlir::TypeAttr attr);
  void setTinAttr(::mlir::ArrayAttr attr);
  void setToutAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTcondAttr();
  ::mlir::Attribute removeTinAttr();
  ::mlir::Attribute removeToutAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::IfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::IfRegionOp declarations
//===----------------------------------------------------------------------===//

class IfRegionOpAdaptor {
public:
  IfRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  IfRegionOpAdaptor(IfRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DictionaryAttr getThenAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getThenAttrs();
  ::mlir::DictionaryAttr getElseAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getElseAttrs();
  ::mlir::tfg::RegionAttr getThenRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getThenRegionAttrs();
  ::mlir::tfg::RegionAttr getElseRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getElseRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class IfRegionOp : public ::mlir::Op<IfRegionOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IfRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("else_attrs"), ::llvm::StringRef("else_region_attrs"), ::llvm::StringRef("then_attrs"), ::llvm::StringRef("then_region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getElseAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getElseAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getElseRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getElseRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getThenAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getThenAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getThenRegionAttrsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getThenRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.IfRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getCond();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::DictionaryAttr getThenAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getThenAttrs();
  ::mlir::DictionaryAttr getElseAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getElseAttrs();
  ::mlir::tfg::RegionAttr getThenRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getThenRegionAttrs();
  ::mlir::tfg::RegionAttr getElseRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getElseRegionAttrs();
  void setThenAttrsAttr(::mlir::DictionaryAttr attr);
  void setElseAttrsAttr(::mlir::DictionaryAttr attr);
  void setThenRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  void setElseRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeThen_attrsAttr();
  ::mlir::Attribute removeElse_attrsAttr();
  ::mlir::Attribute removeThen_region_attrsAttr();
  ::mlir::Attribute removeElse_region_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getThenBlock() { return getThenRegion().front(); }
  Block &getElseBlock() { return getElseRegion().front(); }
  YieldOp getThenYield();
  YieldOp getElseYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (index == 0) return getThenRegionAttrsAttr();
    return getElseRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    if (index == 0) setThenRegionAttrsAttr(attrs);
    else setElseRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::IfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::ReturnOp declarations
//===----------------------------------------------------------------------===//

class ReturnOpAdaptor {
public:
  ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReturnOpAdaptor(ReturnOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getControlRetAttrsAttr();
  ::mlir::ArrayAttr getControlRetAttrs();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReturnOp : public ::mlir::Op<ReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<GraphFuncOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator, ::mlir::OpTrait::IntrinsicOperation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("control_ret_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getControlRetAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getControlRetAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.return");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::ArrayAttr getControlRetAttrsAttr();
  ::mlir::ArrayAttr getControlRetAttrs();
  void setControlRetAttrsAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ValueRange control_operands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::mlir::ArrayAttr control_ret_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::ArrayAttr control_ret_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::ReturnOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulCaseOp declarations
//===----------------------------------------------------------------------===//

class StatefulCaseOpAdaptor {
public:
  StatefulCaseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatefulCaseOpAdaptor(StatefulCaseOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getBranchesAttr();
  ::mlir::ArrayAttr getBranches();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatefulCaseOp : public ::mlir::Op<StatefulCaseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatefulCaseOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tin"), ::llvm::StringRef("Tout"), ::llvm::StringRef("branches"), ::llvm::StringRef("output_shapes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTinAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getToutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getToutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBranchesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBranchesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatefulCase");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getBranchIndexMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::ArrayAttr getBranchesAttr();
  ::mlir::ArrayAttr getBranches();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setBranchesAttr(::mlir::ArrayAttr attr);
  void setTinAttr(::mlir::ArrayAttr attr);
  void setToutAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTinAttr();
  ::mlir::Attribute removeToutAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulCaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulCaseRegionOp declarations
//===----------------------------------------------------------------------===//

class StatefulCaseRegionOpAdaptor {
public:
  StatefulCaseRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatefulCaseRegionOpAdaptor(StatefulCaseRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getBranchAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getBranchAttrs();
  ::mlir::ArrayAttr getRegionAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::RegionRange getBranches();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatefulCaseRegionOp : public ::mlir::Op<StatefulCaseRegionOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatefulCaseRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("branch_attrs"), ::llvm::StringRef("region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBranchAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBranchAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatefulCaseRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getBranchIndex();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getBranchIndexMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::MutableArrayRef<::mlir::Region> getBranches();
  ::mlir::ArrayAttr getBranchAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getBranchAttrs();
  ::mlir::ArrayAttr getRegionAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getRegionAttrs();
  void setBranchAttrsAttr(::mlir::ArrayAttr attr);
  void setRegionAttrsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeBranch_attrsAttr();
  ::mlir::Attribute removeRegion_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getBranchBlock(unsigned idx) { return getBranches()[idx].front(); }
  YieldOp branch_yield(unsigned idx);

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (auto attrs = getRegionAttrsAttr())
      return attrs[index].cast<RegionAttr>();
    return {};
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    SmallVector<Attribute> array = llvm::to_vector(getRegionAttrsAttr());
    array[index] = attrs;
    setRegionAttrsAttr(ArrayAttr::get(getContext(), array));
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulCaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulIfOp declarations
//===----------------------------------------------------------------------===//

class StatefulIfOpAdaptor {
public:
  StatefulIfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatefulIfOpAdaptor(StatefulIfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getThenBranchAttr();
  ::mlir::tf_type::FuncAttr getThenBranch();
  ::mlir::tf_type::FuncAttr getElseBranchAttr();
  ::mlir::tf_type::FuncAttr getElseBranch();
  ::mlir::TypeAttr getTcondAttr();
  ::llvm::Optional<::mlir::Type> getTcond();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatefulIfOp : public ::mlir::Op<StatefulIfOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatefulIfOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tcond"), ::llvm::StringRef("Tin"), ::llvm::StringRef("Tout"), ::llvm::StringRef("else_branch"), ::llvm::StringRef("output_shapes"), ::llvm::StringRef("then_branch")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTcondAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTcondAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTinAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getToutAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getToutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getElseBranchAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getElseBranchAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getThenBranchAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getThenBranchAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatefulIf");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getThenBranchAttr();
  ::mlir::tf_type::FuncAttr getThenBranch();
  ::mlir::tf_type::FuncAttr getElseBranchAttr();
  ::mlir::tf_type::FuncAttr getElseBranch();
  ::mlir::TypeAttr getTcondAttr();
  ::llvm::Optional<::mlir::Type> getTcond();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setThenBranchAttr(::mlir::tf_type::FuncAttr attr);
  void setElseBranchAttr(::mlir::tf_type::FuncAttr attr);
  void setTcondAttr(::mlir::TypeAttr attr);
  void setTinAttr(::mlir::ArrayAttr attr);
  void setToutAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTcondAttr();
  ::mlir::Attribute removeTinAttr();
  ::mlir::Attribute removeToutAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulIfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulIfRegionOp declarations
//===----------------------------------------------------------------------===//

class StatefulIfRegionOpAdaptor {
public:
  StatefulIfRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatefulIfRegionOpAdaptor(StatefulIfRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DictionaryAttr getThenAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getThenAttrs();
  ::mlir::DictionaryAttr getElseAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getElseAttrs();
  ::mlir::tfg::RegionAttr getThenRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getThenRegionAttrs();
  ::mlir::tfg::RegionAttr getElseRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getElseRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatefulIfRegionOp : public ::mlir::Op<StatefulIfRegionOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatefulIfRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("else_attrs"), ::llvm::StringRef("else_region_attrs"), ::llvm::StringRef("then_attrs"), ::llvm::StringRef("then_region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getElseAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getElseAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getElseRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getElseRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getThenAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getThenAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getThenRegionAttrsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getThenRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatefulIfRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getCond();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::DictionaryAttr getThenAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getThenAttrs();
  ::mlir::DictionaryAttr getElseAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getElseAttrs();
  ::mlir::tfg::RegionAttr getThenRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getThenRegionAttrs();
  ::mlir::tfg::RegionAttr getElseRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getElseRegionAttrs();
  void setThenAttrsAttr(::mlir::DictionaryAttr attr);
  void setElseAttrsAttr(::mlir::DictionaryAttr attr);
  void setThenRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  void setElseRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeThen_attrsAttr();
  ::mlir::Attribute removeElse_attrsAttr();
  ::mlir::Attribute removeThen_region_attrsAttr();
  ::mlir::Attribute removeElse_region_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getThenBlock() { return getThenRegion().front(); }
  Block &getElseBlock() { return getElseRegion().front(); }
  YieldOp getThenYield();
  YieldOp getElseYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (index == 0) return getThenRegionAttrsAttr();
    return getElseRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    if (index == 0) setThenRegionAttrsAttr(attrs);
    else setElseRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulIfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulWhileOp declarations
//===----------------------------------------------------------------------===//

class StatefulWhileOpAdaptor {
public:
  StatefulWhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatefulWhileOpAdaptor(StatefulWhileOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getCondAttr();
  ::mlir::tf_type::FuncAttr getCond();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatefulWhileOp : public ::mlir::Op<StatefulWhileOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatefulWhileOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("body"), ::llvm::StringRef("cond"), ::llvm::StringRef("output_shapes"), ::llvm::StringRef("parallel_iterations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getCondAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getCondAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getParallelIterationsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getParallelIterationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatefulWhile");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getCondAttr();
  ::mlir::tf_type::FuncAttr getCond();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setCondAttr(::mlir::tf_type::FuncAttr attr);
  void setBodyAttr(::mlir::tf_type::FuncAttr attr);
  void setParallelIterationsAttr(::mlir::IntegerAttr attr);
  void setTAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulWhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatefulWhileRegionOp declarations
//===----------------------------------------------------------------------===//

class StatefulWhileRegionOpAdaptor {
public:
  StatefulWhileRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatefulWhileRegionOpAdaptor(StatefulWhileRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInit();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::DictionaryAttr getCondAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getCondAttrs();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getCondRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getCondRegionAttrs();
  ::mlir::tfg::RegionAttr getBodyRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getBodyRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getCondRegion();
  ::mlir::Region &getBodyRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatefulWhileRegionOp : public ::mlir::Op<StatefulWhileRegionOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatefulWhileRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("body_attrs"), ::llvm::StringRef("body_region_attrs"), ::llvm::StringRef("cond_attrs"), ::llvm::StringRef("cond_region_attrs"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("parallel_iterations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBodyAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBodyAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getCondAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getCondAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCondRegionAttrsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCondRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getParallelIterationsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getParallelIterationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatefulWhileRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInit();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getInitMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getCondRegion();
  ::mlir::Region &getBodyRegion();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::DictionaryAttr getCondAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getCondAttrs();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getCondRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getCondRegionAttrs();
  ::mlir::tfg::RegionAttr getBodyRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getBodyRegionAttrs();
  void setParallelIterationsAttr(::mlir::IntegerAttr attr);
  void setCondAttrsAttr(::mlir::DictionaryAttr attr);
  void setBodyAttrsAttr(::mlir::DictionaryAttr attr);
  void setCondRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  void setBodyRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeCond_attrsAttr();
  ::mlir::Attribute removeBody_attrsAttr();
  ::mlir::Attribute removeCond_region_attrsAttr();
  ::mlir::Attribute removeBody_region_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  ::mlir::OperandRange getSuccessorEntryOperands(::llvm::Optional<unsigned> index);
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getCondBlock() { return getCondRegion().front(); }
  Block &getBodyBlock() { return getBodyRegion().front(); }
  ConditionOp getCondCondition();
  YieldOp getBodyYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (index == 0) return getCondRegionAttrsAttr();
    return getBodyRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    if (index == 0) setCondRegionAttrsAttr(attrs);
    else setBodyRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatefulWhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessCaseOp declarations
//===----------------------------------------------------------------------===//

class StatelessCaseOpAdaptor {
public:
  StatelessCaseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatelessCaseOpAdaptor(StatelessCaseOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getBranchesAttr();
  ::mlir::ArrayAttr getBranches();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatelessCaseOp : public ::mlir::Op<StatelessCaseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatelessCaseOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tin"), ::llvm::StringRef("Tout"), ::llvm::StringRef("branches"), ::llvm::StringRef("output_shapes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTinAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getToutAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getToutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBranchesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBranchesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatelessCase");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getBranchIndexMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::ArrayAttr getBranchesAttr();
  ::mlir::ArrayAttr getBranches();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setBranchesAttr(::mlir::ArrayAttr attr);
  void setTinAttr(::mlir::ArrayAttr attr);
  void setToutAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTinAttr();
  ::mlir::Attribute removeToutAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value branch_index, ValueRange args, ArrayAttr branches);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange args, ::mlir::ArrayAttr branches, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessCaseOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessCaseRegionOp declarations
//===----------------------------------------------------------------------===//

class StatelessCaseRegionOpAdaptor {
public:
  StatelessCaseRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatelessCaseRegionOpAdaptor(StatelessCaseRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getBranchIndex();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getBranchAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getBranchAttrs();
  ::mlir::ArrayAttr getRegionAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::RegionRange getBranches();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatelessCaseRegionOp : public ::mlir::Op<StatelessCaseRegionOp, ::mlir::OpTrait::VariadicRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatelessCaseRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("branch_attrs"), ::llvm::StringRef("region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBranchAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBranchAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatelessCaseRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getBranchIndex();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getBranchIndexMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::MutableArrayRef<::mlir::Region> getBranches();
  ::mlir::ArrayAttr getBranchAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getBranchAttrs();
  ::mlir::ArrayAttr getRegionAttrsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getRegionAttrs();
  void setBranchAttrsAttr(::mlir::ArrayAttr attr);
  void setRegionAttrsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeBranch_attrsAttr();
  ::mlir::Attribute removeRegion_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value branch_index, ::mlir::ValueRange ctls, /*optional*/::mlir::ArrayAttr branch_attrs, /*optional*/::mlir::ArrayAttr region_attrs, unsigned branchesCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getBranchBlock(unsigned idx) { return getBranches()[idx].front(); }
  YieldOp branch_yield(unsigned idx);

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (auto attrs = getRegionAttrsAttr())
      return attrs[index].cast<RegionAttr>();
    return {};
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    SmallVector<Attribute> array = llvm::to_vector(getRegionAttrsAttr());
    array[index] = attrs;
    setRegionAttrsAttr(ArrayAttr::get(getContext(), array));
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessCaseRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessIfOp declarations
//===----------------------------------------------------------------------===//

class StatelessIfOpAdaptor {
public:
  StatelessIfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatelessIfOpAdaptor(StatelessIfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getThenBranchAttr();
  ::mlir::tf_type::FuncAttr getThenBranch();
  ::mlir::tf_type::FuncAttr getElseBranchAttr();
  ::mlir::tf_type::FuncAttr getElseBranch();
  ::mlir::TypeAttr getTcondAttr();
  ::llvm::Optional<::mlir::Type> getTcond();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatelessIfOp : public ::mlir::Op<StatelessIfOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatelessIfOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("Tcond"), ::llvm::StringRef("Tin"), ::llvm::StringRef("Tout"), ::llvm::StringRef("else_branch"), ::llvm::StringRef("output_shapes"), ::llvm::StringRef("then_branch")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTcondAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTcondAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTinAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTinAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getToutAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getToutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getElseBranchAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getElseBranchAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getThenBranchAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getThenBranchAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatelessIf");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getThenBranchAttr();
  ::mlir::tf_type::FuncAttr getThenBranch();
  ::mlir::tf_type::FuncAttr getElseBranchAttr();
  ::mlir::tf_type::FuncAttr getElseBranch();
  ::mlir::TypeAttr getTcondAttr();
  ::llvm::Optional<::mlir::Type> getTcond();
  ::mlir::ArrayAttr getTinAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTin();
  ::mlir::ArrayAttr getToutAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getTout();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setThenBranchAttr(::mlir::tf_type::FuncAttr attr);
  void setElseBranchAttr(::mlir::tf_type::FuncAttr attr);
  void setTcondAttr(::mlir::TypeAttr attr);
  void setTinAttr(::mlir::ArrayAttr attr);
  void setToutAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTcondAttr();
  ::mlir::Attribute removeTinAttr();
  ::mlir::Attribute removeToutAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, Value cond, ValueRange args, FuncAttr then_branch, FuncAttr else_branch);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr then_branch, ::mlir::tf_type::FuncAttr else_branch, /*optional*/::mlir::TypeAttr Tcond, /*optional*/::mlir::ArrayAttr Tin, /*optional*/::mlir::ArrayAttr Tout, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessIfOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessIfRegionOp declarations
//===----------------------------------------------------------------------===//

class StatelessIfRegionOpAdaptor {
public:
  StatelessIfRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatelessIfRegionOpAdaptor(StatelessIfRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCond();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DictionaryAttr getThenAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getThenAttrs();
  ::mlir::DictionaryAttr getElseAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getElseAttrs();
  ::mlir::tfg::RegionAttr getThenRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getThenRegionAttrs();
  ::mlir::tfg::RegionAttr getElseRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getElseRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatelessIfRegionOp : public ::mlir::Op<StatelessIfRegionOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatelessIfRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("else_attrs"), ::llvm::StringRef("else_region_attrs"), ::llvm::StringRef("then_attrs"), ::llvm::StringRef("then_region_attrs")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getElseAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getElseAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getElseRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getElseRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getThenAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getThenAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getThenRegionAttrsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getThenRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatelessIfRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getCond();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getCondMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::DictionaryAttr getThenAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getThenAttrs();
  ::mlir::DictionaryAttr getElseAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getElseAttrs();
  ::mlir::tfg::RegionAttr getThenRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getThenRegionAttrs();
  ::mlir::tfg::RegionAttr getElseRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getElseRegionAttrs();
  void setThenAttrsAttr(::mlir::DictionaryAttr attr);
  void setElseAttrsAttr(::mlir::DictionaryAttr attr);
  void setThenRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  void setElseRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeThen_attrsAttr();
  ::mlir::Attribute removeElse_attrsAttr();
  ::mlir::Attribute removeThen_region_attrsAttr();
  ::mlir::Attribute removeElse_region_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::ValueRange ctls, /*optional*/::mlir::DictionaryAttr then_attrs, /*optional*/::mlir::DictionaryAttr else_attrs, /*optional*/::mlir::tfg::RegionAttr then_region_attrs, /*optional*/::mlir::tfg::RegionAttr else_region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getThenBlock() { return getThenRegion().front(); }
  Block &getElseBlock() { return getElseRegion().front(); }
  YieldOp getThenYield();
  YieldOp getElseYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (index == 0) return getThenRegionAttrsAttr();
    return getElseRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    if (index == 0) setThenRegionAttrsAttr(attrs);
    else setElseRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessIfRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessWhileOp declarations
//===----------------------------------------------------------------------===//

class StatelessWhileOpAdaptor {
public:
  StatelessWhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatelessWhileOpAdaptor(StatelessWhileOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getCondAttr();
  ::mlir::tf_type::FuncAttr getCond();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatelessWhileOp : public ::mlir::Op<StatelessWhileOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatelessWhileOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("body"), ::llvm::StringRef("cond"), ::llvm::StringRef("output_shapes"), ::llvm::StringRef("parallel_iterations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getCondAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getCondAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getParallelIterationsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getParallelIterationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatelessWhile");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getCondAttr();
  ::mlir::tf_type::FuncAttr getCond();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setCondAttr(::mlir::tf_type::FuncAttr attr);
  void setBodyAttr(::mlir::tf_type::FuncAttr attr);
  void setParallelIterationsAttr(::mlir::IntegerAttr attr);
  void setTAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessWhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::StatelessWhileRegionOp declarations
//===----------------------------------------------------------------------===//

class StatelessWhileRegionOpAdaptor {
public:
  StatelessWhileRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  StatelessWhileRegionOpAdaptor(StatelessWhileRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInit();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::DictionaryAttr getCondAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getCondAttrs();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getCondRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getCondRegionAttrs();
  ::mlir::tfg::RegionAttr getBodyRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getBodyRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getCondRegion();
  ::mlir::Region &getBodyRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class StatelessWhileRegionOp : public ::mlir::Op<StatelessWhileRegionOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StatelessWhileRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("body_attrs"), ::llvm::StringRef("body_region_attrs"), ::llvm::StringRef("cond_attrs"), ::llvm::StringRef("cond_region_attrs"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("parallel_iterations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBodyAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBodyAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getCondAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getCondAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCondRegionAttrsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCondRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getParallelIterationsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getParallelIterationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.StatelessWhileRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInit();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getInitMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getCondRegion();
  ::mlir::Region &getBodyRegion();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::DictionaryAttr getCondAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getCondAttrs();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getCondRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getCondRegionAttrs();
  ::mlir::tfg::RegionAttr getBodyRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getBodyRegionAttrs();
  void setParallelIterationsAttr(::mlir::IntegerAttr attr);
  void setCondAttrsAttr(::mlir::DictionaryAttr attr);
  void setBodyAttrsAttr(::mlir::DictionaryAttr attr);
  void setCondRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  void setBodyRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeCond_attrsAttr();
  ::mlir::Attribute removeBody_attrsAttr();
  ::mlir::Attribute removeCond_region_attrsAttr();
  ::mlir::Attribute removeBody_region_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  ::mlir::OperandRange getSuccessorEntryOperands(::llvm::Optional<unsigned> index);
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getCondBlock() { return getCondRegion().front(); }
  Block &getBodyBlock() { return getBodyRegion().front(); }
  ConditionOp getCondCondition();
  YieldOp getBodyYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (index == 0) return getCondRegionAttrsAttr();
    return getBodyRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    if (index == 0) setCondRegionAttrsAttr(attrs);
    else setBodyRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::StatelessWhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::WhileOp declarations
//===----------------------------------------------------------------------===//

class WhileOpAdaptor {
public:
  WhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WhileOpAdaptor(WhileOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::tf_type::FuncAttr getCondAttr();
  ::mlir::tf_type::FuncAttr getCond();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class WhileOp : public ::mlir::Op<WhileOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolUserOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WhileOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("T"), ::llvm::StringRef("body"), ::llvm::StringRef("cond"), ::llvm::StringRef("output_shapes"), ::llvm::StringRef("parallel_iterations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getCondAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getCondAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getOutputShapesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getOutputShapesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getParallelIterationsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getParallelIterationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.While");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::tf_type::FuncAttr getCondAttr();
  ::mlir::tf_type::FuncAttr getCond();
  ::mlir::tf_type::FuncAttr getBodyAttr();
  ::mlir::tf_type::FuncAttr getBody();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::ArrayAttr getTAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getT();
  ::mlir::ArrayAttr getOutputShapesAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOutputShapes();
  void setCondAttr(::mlir::tf_type::FuncAttr attr);
  void setBodyAttr(::mlir::tf_type::FuncAttr attr);
  void setParallelIterationsAttr(::mlir::IntegerAttr attr);
  void setTAttr(::mlir::ArrayAttr attr);
  void setOutputShapesAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeTAttr();
  ::mlir::Attribute removeOutput_shapesAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, ValueRange args, FuncAttr cond, FuncAttr body, IntegerAttr parallel_iterations);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::tf_type::FuncAttr cond, ::mlir::tf_type::FuncAttr body, uint64_t parallel_iterations, /*optional*/::mlir::ArrayAttr T, /*optional*/::mlir::ArrayAttr output_shapes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::WhileOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::WhileRegionOp declarations
//===----------------------------------------------------------------------===//

class WhileRegionOpAdaptor {
public:
  WhileRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WhileRegionOpAdaptor(WhileRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInit();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::DictionaryAttr getCondAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getCondAttrs();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getCondRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getCondRegionAttrs();
  ::mlir::tfg::RegionAttr getBodyRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getBodyRegionAttrs();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getCondRegion();
  ::mlir::Region &getBodyRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class WhileRegionOp : public ::mlir::Op<WhileRegionOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasOnlyGraphRegion, ::mlir::RegionKindInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::tfg::PreservedAttributesInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::tfg::ControlArgumentInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WhileRegionOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("body_attrs"), ::llvm::StringRef("body_region_attrs"), ::llvm::StringRef("cond_attrs"), ::llvm::StringRef("cond_region_attrs"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("parallel_iterations")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBodyAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBodyAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBodyRegionAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBodyRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getCondAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getCondAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCondRegionAttrsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCondRegionAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getParallelIterationsAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getParallelIterationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.WhileRegion");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInit();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getInitMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getOuts();
  ::mlir::TypedValue<::mlir::tf_type::ControlType> getCtl();
  ::mlir::Region &getCondRegion();
  ::mlir::Region &getBodyRegion();
  ::mlir::IntegerAttr getParallelIterationsAttr();
  uint64_t getParallelIterations();
  ::mlir::DictionaryAttr getCondAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getCondAttrs();
  ::mlir::DictionaryAttr getBodyAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getBodyAttrs();
  ::mlir::tfg::RegionAttr getCondRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getCondRegionAttrs();
  ::mlir::tfg::RegionAttr getBodyRegionAttrsAttr();
  ::llvm::Optional<::mlir::tfg::RegionAttr> getBodyRegionAttrs();
  void setParallelIterationsAttr(::mlir::IntegerAttr attr);
  void setCondAttrsAttr(::mlir::DictionaryAttr attr);
  void setBodyAttrsAttr(::mlir::DictionaryAttr attr);
  void setCondRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  void setBodyRegionAttrsAttr(::mlir::tfg::RegionAttr attr);
  ::mlir::Attribute removeCond_attrsAttr();
  ::mlir::Attribute removeBody_attrsAttr();
  ::mlir::Attribute removeCond_region_attrsAttr();
  ::mlir::Attribute removeBody_region_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, ::mlir::IntegerAttr parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outs, ::mlir::Type ctl, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange init, ::mlir::ValueRange ctls, uint64_t parallel_iterations, /*optional*/::mlir::DictionaryAttr cond_attrs, /*optional*/::mlir::DictionaryAttr body_attrs, /*optional*/::mlir::tfg::RegionAttr cond_region_attrs, /*optional*/::mlir::tfg::RegionAttr body_region_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::llvm::StringRef getDefaultDialect();
  ::mlir::OperandRange getSuccessorEntryOperands(::llvm::Optional<unsigned> index);
  void getSuccessorRegions(::llvm::Optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  static mlir::BlockArgument getDataValueOf(BlockArgument ctl);
  static mlir::BlockArgument getControlTokenOf(BlockArgument data);
  static mlir::BlockArgument getDataValue(Region &region, unsigned idx);
  static mlir::BlockArgument getControlToken(Region &region, unsigned idx);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  Block &getCondBlock() { return getCondRegion().front(); }
  Block &getBodyBlock() { return getBodyRegion().front(); }
  ConditionOp getCondCondition();
  YieldOp getBodyYield();

  bool areTypesCompatible(Type lhs, Type rhs) {
    return tf_type::HasCompatibleElementTypes(lhs, rhs);
  }

  RegionAttr getPreservedAttrs(unsigned index) {
    if (index == 0) return getCondRegionAttrsAttr();
    return getBodyRegionAttrsAttr();
  }
  void setPreservedAttrs(unsigned index, RegionAttr attrs) {
    if (index == 0) setCondRegionAttrsAttr(attrs);
    else setBodyRegionAttrsAttr(attrs);
  }
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::WhileRegionOp)

namespace mlir {
namespace tfg {

//===----------------------------------------------------------------------===//
// ::mlir::tfg::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getArgs();
  ::mlir::ValueRange getCtls();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator, ::mlir::RegionBranchTerminatorOpInterface::Trait, ::mlir::OpTrait::IntrinsicOperation> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tfg.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getArgs();
  ::mlir::Operation::operand_range getCtls();
  ::mlir::MutableOperandRange getArgsMutable();
  ::mlir::MutableOperandRange getCtlsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args, ::mlir::ValueRange ctls);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange args, ::mlir::ValueRange ctls);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::llvm::Optional<unsigned> index);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace tfg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tfg::YieldOp)


#endif  // GET_OP_CLASSES

