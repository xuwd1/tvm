#ifndef TVM_TSL_TE_IVAR_VISITOR_H_
#define TVM_TSL_TE_IVAR_VISITOR_H_

namespace tvm {
namespace te {
TVM_DLL void ExtractRootPathIvarShape(const ComputeOp& op, const Array<IterVar>& requested,
                        std::unordered_map<IterVar, PrimExpr>* out_shape_map);

}
}



#endif