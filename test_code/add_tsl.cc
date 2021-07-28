#include <tvm/te/operation.h>
#include <tvm/tir/var.h>
// #include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/te/schedule_pass.h>
#include <tvm/tsl/tir/expr.h>
// #include <tvm/tir/buffer.h>

#include <tvm/tir/stmt_functor.h>

// #include <tvm/tir/expr.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/c_runtime_api.h>

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

int main() {
  const int M = 512;
  const int N = 256;
  te::Tensor A = te::placeholder({M, N}, DataType::Float(32), "A");
  te::Tensor B = te::placeholder({M, N}, DataType::Float(32), "B");
  cout << A << endl;
  cout << B << endl;

  te::Tensor C = te::compute(
      {M, N}, std::function<te::TslExpr(tir::Var, tir::Var)>([=](tir::Var i, tir::Var j) {
        return te::TslAdd(A.TslPLoad({i, j}), B.TslPLoad({i, j}));
      }));

  auto C_op = C->op;
  auto C_computeNode = *C_op.as<te::ComputeOpNode>();
  cout << C_computeNode.in_eshape << endl;
  cout << C_computeNode.in_ushape << endl;
  cout << C_computeNode.out_eshape << endl;
  cout << C_computeNode.out_ushape << endl;
  cout << C_computeNode.input_elemshape(0) << endl;
  cout << C_computeNode.input_unionshape(0) << endl;
  cout << C_computeNode.output_elemshape(0) << endl;
  cout << C_computeNode.output_unionshape(0) << endl;

  cout << C << endl;
}