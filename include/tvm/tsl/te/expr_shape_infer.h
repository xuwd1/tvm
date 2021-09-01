#ifndef TVM_TSL_TE_EXPR_SHAPE_INFER_H_
#define TVM_TSL_TE_EXPR_SHAPE_INFER_H_

#include <unordered_map>
#include <tvm/runtime/object.h>


namespace tvm {
namespace te {
TVM_DLL std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> CollectDim(TslExpr e);

TVM_DLL std::unordered_map<const TslExprNode*, Array<PrimExpr>> InferShape(const Stage& stage);

}
}


#endif