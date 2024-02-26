/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tfg::TFGraphDialect)
namespace mlir {
namespace tfg {

TFGraphDialect::TFGraphDialect(::mlir::MLIRContext *context) 
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<TFGraphDialect>()) {
  
  initialize();
}
} // namespace tfg
} // namespace mlir
