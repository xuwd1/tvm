#ifndef TVM_TSL_TE_EXPR_SHAPE_INFER_H_
#define TVM_TSL_TE_EXPR_SHAPE_INFER_H_

namespace tvm {
namespace te {
TVM_DLL std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> CollectDim(TslExpr e);
}
}


#endif