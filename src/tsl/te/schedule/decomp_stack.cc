#include <tvm/te/schedule.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

namespace tvm {
namespace te {




auto StageNode::DecompEntry::Create(PrimExpr factor, IterVar pathivar) -> DecompEntry {
  DecompEntry ret;
  ret.factor = factor;
  ret.pathivar = pathivar;
  ret.leaf_vars.push_back(pathivar);
  ret.all_vars.push_back(pathivar);
  return ret;
}

auto StageNode::DecompStack::Create(IterVarType iter_type)->DecompStack {
  DecompStack ret;
  ret.iter_type=iter_type;
  return ret;
}
auto StageNode::DecompStack::size() const ->size_t {
  return this->entries.size();
}

auto StageNode::DecompStack::operator[](size_t index) ->StageNode::DecompEntry& {
  return this->entries[index];
}


}
}