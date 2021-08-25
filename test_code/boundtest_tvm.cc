#include <tvm/te/operation.h>
#include <tvm/tir/var.h>
// #include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tsl/tir/expr.h>
// #include <tvm/tir/buffer.h>
#include <tvm/driver/driver_api.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/var.h>
#include <tvm/tsl/tir/op.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tvm;
using namespace std;

int main() {
  te::Tensor A = te::placeholder({18});
  te::Tensor B = te::compute({18}, [=](tir::Var i) { return A[i]; }, "B");
  te::Tensor C = te::compute({18}, [=](tir::Var i) { return B[i]; }, "C");
  auto sch = te::create_schedule({C->op});
  auto ax = C->op.as<te::ComputeOpNode>()->axis;
  tir::IterVar outer, inner;
  sch[C].split(ax[0], 16, &outer, &inner);
  sch[C].split(inner, 4, &outer, &inner);

  sch[B].compute_at(sch[C], outer);
  auto args = Array<te::Tensor>({A, B, C});
  std::unordered_map<te::Tensor, te::Buffer> binds;
  auto target = Target("llvm");
  /*auto bounds = te::InferBound(sch);
  for (auto& v : sch[C]->leaf_iter_vars) {
    cout << v << ":" << bounds[v] << endl;
  }
  cout << "B:" << endl;
  for (auto& v : sch[B]->leaf_iter_vars) {
    cout << v << ":" << bounds[v] << endl;
  }*/
  //cout << bounds << endl;
  auto lowered = lower(sch, args, "func", binds);
  cout << lowered << endl;
}