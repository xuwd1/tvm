#include <tvm/te/operation.h>
#include <tvm/tir/var.h>
// #include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tsl/tir/expr.h>
// #include <tvm/tir/buffer.h>
#include <tvm/tsl/tir/op.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/var.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tvm;
using namespace std;

void printTensorAttrs(const te::Tensor& A) {
  cout << A->GetNameHint() << endl;
  cout << A->write_eshape << endl;
  cout << A->write_ushape << endl;
  cout << A->read_eshape << endl;
  cout << A->read_ushape << endl;
}

void printDecompDBGInfo(te::StageNode* stage) {
  cout << "DECOMP STACK for " << stage << endl;
  auto& stack = stage->decomp_stack;
  for (size_t i = 0; i < stack.size(); i++) {
    cout << "STACK LEVEL " << i << ":\n";
    for (auto& v : stack[i].split_relations) {
      cout << v << endl;
    }
  }
  cout << "LEAF ITERVARS:" << endl;
  cout << stage->leaf_iter_vars << endl;
}

int main() {
  const int M = 512;
  const int N = 256;
  const int K = 128;

  te::Tensor A = te::Tslplaceholder({M, K}, DataType::Float(32), "A");
  te::Tensor B = te::Tslplaceholder({K, N}, DataType::Float(32), "B");
  auto r = te::reduce_axis(Range(0, 1), "r");
  te::Tensor C = te::compute(
      {M, N}, std::function<tir::TslExpr(tir::Var, tir::Var)>([=](tir::Var i, tir::Var j) { 
        return tsl_sum(tir::TslGemm(A.TslPLoad({i, r}), B.TslPLoad({r, j})), {r});
      }),
      "tslgemm(A,B)");

  auto sch = te::create_schedule({C->op});
  cout << sch->stages << endl;
  

  // printDecompDBGInfo(sch[D].operator->());
}