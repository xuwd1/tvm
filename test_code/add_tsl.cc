#include <tvm/te/operation.h>
#include <tvm/tir/var.h>
// #include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tsl/tir/expr.h>
// #include <tvm/tir/buffer.h>

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

  te::Tensor A = te::Tslplaceholder({M, N}, DataType::Float(32), "A");
  te::Tensor B = te::Tslplaceholder({M, N}, DataType::Float(32), "B");
  te::Tensor X = te::Tslplaceholder({M, N}, DataType::Float(32), "X");
  te::Tensor Y = te::Tslplaceholder({M, N}, DataType::Float(32), "Y");
  cout << A << endl;
  cout << B << endl;

  te::Tensor C = te::compute(
      {M, N}, std::function<te::TslExpr(tir::Var, tir::Var)>([=](tir::Var i, tir::Var j) {
        return te::TslAdd(A.TslPLoad({i, j}), B.TslPLoad({i, j}));
      }),
      "tsladd(A,B)");

  // auto C_op = C->op;
  // auto C_computeNode = *C_op.as<te::ComputeOpNode>();
  // cout << C_computeNode.in_eshape << endl;
  // cout << C_computeNode.in_ushape << endl;
  // cout << C_computeNode.out_eshape << endl;
  // cout << C_computeNode.out_ushape << endl;
  // cout << C_computeNode.input_elemshape(0) << endl;
  // cout << C_computeNode.input_unionshape(0) << endl;
  // cout << C_computeNode.output_elemshape(0) << endl;
  // cout << C_computeNode.output_unionshape(0) << endl;

  // cout << C << endl;
  // cout << C->op->InputTensors() << endl;

  te::Tensor D = te::compute(
      {M, N}, std::function<te::TslExpr(tir::Var, tir::Var)>([=](tir::Var i, tir::Var j) {
        return te::TslAdd(C.TslPLoad({i, j}), X.TslPLoad({i, j}));
      }),
      "tsladd(X,(A+B))");
  te::Tensor E = te::compute(
      {M, N}, std::function<te::TslExpr(tir::Var, tir::Var)>([=](tir::Var i, tir::Var j) {
        return te::TslAdd(D.TslPLoad({i, j}), Y.TslPLoad({i, j}));
      }),
      "tsladd(Y,(A+B+X))");



  auto sch = te::create_schedule({E->op});
  cout << sch->stages << endl;
  // printDecompDBGInfo(sch[D].operator->());
  Array<tir::IterVar> ret1;
  sch[D].decompose({32, 32}, ret1);
  cout << ret1 << endl;
      // printDecompDBGInfo(sch[D].operator->());
  Array<tir::IterVar> ret2;
  sch[D].decompose({16, 16},ret2);
  cout << ret2 << endl;
  // printDecompDBGInfo(sch[D].operator->());
}