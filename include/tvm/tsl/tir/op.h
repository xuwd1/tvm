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

TVM_DLL TslExpr gemm(tir::TslExpr src, Array<tir::IterVar> axis, Array<tir::TslExpr> init = {});



}


#endif //TVM_TSL_TIR_OP_H_