#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace te {

size_t FindVar(const Array<IterVar>& array, const Var& v) {
  const Object* ptr = v.get();
  for (size_t i = 0; i < array.size(); i++) {
    if (array[i]->var.get() == ptr) {
      return i;
    }
  }
  return array.size();
}

void ExtractRootPathIvarShape(const ComputeOp& op, const Array<IterVar>& requested,
                        std::unordered_map<IterVar, PrimExpr>* out_shape_map) {
  auto& map = *out_shape_map;
  auto fvisit = [&](const ObjectRef& n) {
    if (auto* pload = n.as<TslProducerLoadNode>()) {
      
      Array<Array<PrimExpr>> c_indices = pload->c_indices;
      Tensor t = Downcast<Tensor>(pload->producer);
      CHECK(t->op.defined());
      const Operation& producer_op = t->op;
      Array<PrimExpr> shape = producer_op->output_shape(t->value_index);
      CHECK(c_indices.size() == t.ndim());
      for (size_t i = 0; i < c_indices.size(); i++) {
        Array<PrimExpr> c_index = c_indices[i];
        const VarNode* index_var;
        if (c_index.size()==1) {
          index_var=c_index[0].as<VarNode>();
          CHECK(index_var != nullptr);
          size_t pos = FindVar(requested, GetRef<Var>(index_var));
          if (pos < requested.size()) {
            if (map.count(requested[pos]) != 0) {  // indexer occurred more than once
              auto x = map[requested[pos]];
              CHECK((x == shape[i]).as<tir::IntImmNode>()->value);  
            } else {
              map[requested[pos]] = shape[i];
            }
          }
        }
      }
    }
  };
  for (const auto& e:op->body) {
    tir::PostOrderVisit(e,fvisit);
  }
}
}
}