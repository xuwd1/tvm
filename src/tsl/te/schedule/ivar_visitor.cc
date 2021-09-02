#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include "cind_utility.h"
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
  Array<IterVar> reduce_ivars;
  const auto self_shape=op->output_shape(0);
  const auto self_ivars=op->root_iter_vars();
  for (auto& iv:requested) {
    if (iv->iter_type==IterVarType::kDataPar) {
      size_t pos=FindVar(self_ivars,iv->var);
      CHECK(pos<self_ivars.size());
      map[iv]=self_shape[pos];
    }else {
      reduce_ivars.push_back(iv);
    }
  }
  auto fvisit = [&](const ObjectRef& n) {
    if (auto* pload = n.as<TslProducerLoadNode>()) {
      Array<Array<PrimExpr>> c_indices = pload->c_indices;
      Tensor t = Downcast<Tensor>(pload->producer);
      CHECK(t->op.defined());
      const Operation& producer_op = t->op;
      Array<PrimExpr> producer_shape = producer_op->output_shape(t->value_index);
      CHECK(c_indices.size() == t.ndim());

      for (auto& iv:reduce_ivars) {
        for (size_t i=0;i<c_indices.size();i++) {
          auto& c_index=c_indices[i];
          if (IsPureCIndex(c_index)&&Downcast<Var>(c_index[0]).get()==iv->var.get()) {
            map[iv]=producer_shape[i];
          }
        }
      }


      /*
      for (size_t i = 0; i < c_indices.size(); i++) {
        Array<PrimExpr> c_index = c_indices[i];
        if (IsPureCIndex(c_index)) { 
          auto& index_var=c_index[0];
          size_t pos = FindVar(requested, Downcast<Var>(index_var));
          if (pos < requested.size()) {
            if (map.count(requested[pos]) != 0) {  // indexer occurred more than once
              auto x = map[requested[pos]];
              CHECK((x == shape[i]).as<tir::IntImmNode>()->value);  
            } else {
              map[requested[pos]] = shape[i];
            }
          }else { 
            LOG_FATAL;
          }
        }else {
          for (auto& index_var:c_index) {
            size_t pos=FindVar(requested,Downcast<Var>(index_var));
            if (pos< requested.size()) {
              auto& iv = requested[pos];
              if (!map.count(iv)) {
                map[iv]=shape[i];
              }
            }else {
              LOG_FATAL;
            }
          }
        }
      }
      */
    }
  };
  for (const auto& e:op->body) {
    tir::PostOrderVisit(e,fvisit);
  }
}
}
}