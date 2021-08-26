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
      
      Array<PrimExpr> indices = pload->indices; //TODO:indices should be modified to support compound indexing, which is supposed to be used for supporting CONV
      Tensor t = Downcast<Tensor>(pload->producer);
      CHECK(t->op.defined());
      const Operation& producer_op = t->op;
      Array<PrimExpr> shape = producer_op->output_shape(t->value_index);
      CHECK(indices.size() == t.ndim());
      for (size_t i = 0; i < indices.size(); i++) { 
        const auto index_var=indices[i].as<VarNode>(); //TODO:compound indexing also needs this modified
        CHECK(index_var!=nullptr);
        size_t pos = FindVar(requested, GetRef<Var>(index_var));
        if (pos  < requested.size()) {
          if (map.count(requested[pos])!=0) { //indexer occurred more than once
            auto x = map[requested[pos]];
            CHECK((x==shape[i]).as<tir::IntImmNode>()->value); //TODO:in compound indexing, only gather pure indexing if a indexer occurred more than once
          } else {
            map[requested[pos]] = shape[i];
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