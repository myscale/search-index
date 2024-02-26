/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Definitions                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/// Given a block argument that is a control token, return the associated
/// data argument.
mlir::BlockArgument mlir::tfg::ControlArgumentInterface::getDataValueOf(BlockArgument ctl) {
      return getImpl()->getDataValueOf(ctl);
  }
/// Given a block argument that is a data value, return the associated
/// control token.
mlir::BlockArgument mlir::tfg::ControlArgumentInterface::getControlTokenOf(BlockArgument data) {
      return getImpl()->getControlTokenOf(data);
  }
/// Get the data value corresponding to a given argument index.
mlir::BlockArgument mlir::tfg::ControlArgumentInterface::getDataValue(Region & region, unsigned idx) {
      return getImpl()->getDataValue(region, idx);
  }
/// Get the control token corresponding to a given argument index.
mlir::BlockArgument mlir::tfg::ControlArgumentInterface::getControlToken(Region & region, unsigned idx) {
      return getImpl()->getControlToken(region, idx);
  }
/// Get the (potentially null) RegionAttr corresponding to the region
/// at the given index.
mlir::tfg::RegionAttr mlir::tfg::PreservedAttributesInterface::getPreservedAttrs(unsigned idx) {
      return getImpl()->getPreservedAttrs(getImpl(), getOperation(), idx);
  }
/// Set the non-null RegionAttr corresponding to the region at the
/// given index.
void mlir::tfg::PreservedAttributesInterface::setPreservedAttrs(unsigned idx, RegionAttr attrs) {
      return getImpl()->setPreservedAttrs(getImpl(), getOperation(), idx, attrs);
  }
/// Returns true if the current op is stateful, according to TensorFlow.
bool mlir::tfg::TensorFlowRegistryInterface::isStateful() {
      return getImpl()->isStateful(getImpl(), getOperation());
  }
