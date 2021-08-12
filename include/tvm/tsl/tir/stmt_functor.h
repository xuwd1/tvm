#ifndef TVM_TIR_STMT_FUNCTOR_H_
#define TVM_TIR_STMT_FUNCTOR_H_

#include <tvm/node/container.h>
#include <tvm/node/functor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/tsl/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/stmt.h>

#include <unordered_map>
#include <utility>

namespace tvm {
namespace tir {

  /*!
 * \brief Substitute the var specified by vmap.
 * \param stmt The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 * \return The converted form.
 */

//TODO: implement this when things come to stmt
//TVM_DLL Stmt Substitute(Stmt stmt, std::function<Optional<PrimExpr>(const Var& var)> vmap);

/*!
 * \brief Substitute the var specified by vmap.
 * \param expr The source statement to be substituted
 * \param vmap returns a new value if re-mapping is needed, otherwise returns nullptr.
 * \return The result.
 */
TVM_DLL TslExpr Substitute(TslExpr expr, std::function<Optional<TslExpr>(const TslVar& var)> vmap);

/*!
 * \brief Sugar for substitute via a given map.
 * \param input The input to be updated.
 * \param value_map The map of new values.
 * \return The result.
 * \tparam T the input type, can be PrimExpr or Stmt.
 */
template <typename T>
inline auto Substitute(T input, const Map<TslVar, TslExpr>& value_map) {
  auto vmap = [&](const TslVar& var) -> Optional<TslExpr> {
    auto it = value_map.find(var);
    if (it != value_map.end()) return (*it).second;
    return Optional<TslExpr>(nullptr);
  };
  return Substitute(std::move(input), vmap);
}

/*!
 * \brief Sugar for substitute via a given map.
 * \param input The input to be updated.
 * \param value_map The map of new values.
 * \return The result.
 * \tparam T the input type, can be PrimExpr or Stmt.
 */
template <typename T>
inline T Substitute(T input, const std::unordered_map<const TslVarNode*, TslExpr>& value_map) {
  auto vmap = [&](const TslVar& var) -> Optional<TslExpr> {
    auto it = value_map.find(var.get());
    if (it != value_map.end()) return (*it).second;
    return Optional<TslExpr>(nullptr);
  };
  return Substitute(std::move(input), vmap);
}

}
}
#endif