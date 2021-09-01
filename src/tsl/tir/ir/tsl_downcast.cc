#include <tvm/tir/expr_functor.h>
#include <tvm/tsl/tir/tsl_downcast.h>
#include <tvm/tir/functor_common.h>


namespace tvm {
namespace tir {

#define DEFINE_TSL_RUNTIME_DOWNCAST_ENTRY(ARG, OP) \
if (ARG->IsInstance<OP##Node>()) {               \
  return Downcast<OP>(ARG);                      \
}


TslExpr TslRuntimeDowncast(const PrimExpr& e) {
  DEFINE_TSL_RUNTIME_DOWNCAST_ENTRY(e, TslAdd);
  DEFINE_TSL_RUNTIME_DOWNCAST_ENTRY(e, TslProducerLoad);
  DEFINE_TSL_RUNTIME_DOWNCAST_ENTRY(e, TslReduce);
}

}
}