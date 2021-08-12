#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/tsl/stmt_functor.h>

#include <functional>
#include <tvm/tir/functor_common.h>

namespace tvm {
namespace tir {



// Mutator for substituting TslVar to TslExpr, just like what TVM does
class TslIRSubstitue : public StmtExprMutator {
 public:
  explicit TslIRSubstitue(std::function<Optional<TslExpr>(const TslVar&)> vmap) : vmap_(vmap) {}

  PrimExpr VisitExpr_(const TslVarNode* op) final {
    TslVar var = GetRef<TslVar>(op);
    auto ret = vmap_(var);
    if (ret.defined()) return ret.value();
    return std::move(var);
  }

  /* //TODO: finish this when things comes to 
  PrimExpr VisitExpr_(const LoadNode* op) final {
    // NOTE: we do not explicit recursivly mutate op->buffer_var
    PrimExpr ret = StmtExprMutator::VisitExpr_(op);
    op = ret.as<LoadNode>();
    if (auto mapped_var = vmap_(op->buffer_var)) {
      return Load(op->dtype, Downcast<Var>(mapped_var.value()), op->index, op->predicate);
    } else {
      return ret;
    }
  }

  Stmt VisitStmt_(const StoreNode* op) final {
    // NOTE: we do not explicit recursivly mutate op->buffer_var
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    op = ret.as<StoreNode>();
    if (auto mapped_var = vmap_(op->buffer_var)) {
      return Store(Downcast<Var>(mapped_var.value()), op->value, op->index, op->predicate);
    } else {
      return ret;
    }
  }
  */

 private:
  std::function<Optional<TslExpr>(const TslVar&)> vmap_;
};

TslExpr Substitute(TslExpr expr, std::function<Optional<TslExpr>(const TslVar& var)> vmap) {
  return TslIRSubstitue(vmap)(std::move(expr));
}


} 
}  // namespace tvm