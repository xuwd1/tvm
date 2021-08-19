#ifndef TVM_TSL_TIR_OP_H_
#define TVM_TSL_TIR_OP_H_

#include <tvm/ir/op.h>
#include <tvm/ir/type.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>
#include <tvm/tsl/tir/expr.h>

#include <algorithm>
#include <limits>
#include <type_traits>

namespace tvm {
namespace tir {



TVM_DLL TslExpr gemm(tir::TslExpr src, Array<tir::IterVar> axis, Array<tir::TslExpr> init = {});

inline TslExpr make_tslzero(DataType t) { 
  CHECK(t.is_scalar());
  CHECK(!t.is_handle());
  if (t.is_int()) {
    return TslIntImm(t, static_cast<int64_t>(0));
  } 
  if (t.is_float()) {
    return TslFloatImm(t, static_cast<double>(0));
  }
  LOG(FATAL) << "cannot make tslzero for " << t;
  return TslExpr();
}
}  // namespace tir
}


#endif //TVM_TSL_TIR_OP_H_