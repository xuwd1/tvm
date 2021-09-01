#ifndef TVM_TSL_TIR_OP_H_
#define TVM_TSL_TIR_OP_H_


#include <tvm/node/functor.h>
#include <tvm/tir/expr.h>
#include <tvm/tsl/tir/expr.h>
#include <utility>


namespace tvm {
namespace tir {

TVM_DLL TslExpr TslRuntimeDowncast(const PrimExpr& e);

}
}

#endif