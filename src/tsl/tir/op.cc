#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tsl/tir/expr.h>
#include <tvm/tsl/tir/op.h>
#include <tvm/tir/op.h>

#include <cmath>

namespace tvm {
using namespace tir;

TslExpr gemm(tir::TslExpr src, Array<tir::IterVar> axis, Array<tir::TslExpr> init) {
  CHECK(src.dtype().lanes() == 1);
  TslVar x("x", src.dtype()), y("y", src.dtype());
  TslExpr result = TslAdd(x, y);
  TslExpr identity = make_tslzero(src.dtype());
  TslCommReducer combiner = TslCommReducer({x}, {y}, {result}, {identity});
  return TslReduce(combiner, {src}, axis, make_const(DataType::Bool(1),true) , 0, init);
}

}  // namespace tvm