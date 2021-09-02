#ifndef TVM_TSL_CIND_UTILITY_H_
#define TVM_TSL_CIND_UTILITY_H_


namespace tvm {
namespace te {
bool IsPureCIndex(const Array<PrimExpr>& c_index);

bool IsCIndexEqual(const Array<PrimExpr>& ca, const Array<PrimExpr>& cb);

size_t FindCIndex(const Array<PrimExpr>& c_index, const Array<Array<PrimExpr>>& target_c_indices);

bool IsCIndicesEqual(const Array<Array<PrimExpr>>& ca, const Array<Array<PrimExpr>>& cb);
}
}




#endif