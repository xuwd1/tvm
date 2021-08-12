#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tsl/tir/expr.h>
#include <tvm/tsl/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include <cmath>


namespace tvm {
using namespace tir;


TslExpr gemm(tir::TslExpr src, Array<tir::IterVar> axis, Array<tir::TslExpr> init) {
  Var x("x", src.dtype()), y("y", src.dtype);
  TslExpr result = TslAdd(x, y);
  var;

}

}  // namespace tvm