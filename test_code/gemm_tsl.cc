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
#include <tvm/tsl/tir/op.h>
#include <tvm/tsl/te/expr_shape_infer.h>
#include <tvm/tsl/tsl_debug.h>
#include <tvm/tsl/tsl_debug_lang.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace tvm;
using namespace std;

int main() {
  const int M = 512;
  const int N = 256;
  const int K = 128;

  te::Tensor A = te::Tslplaceholder({M, K}, DataType::Float(32), "A");
  te::Tensor B = te::Tslplaceholder({K, N}, DataType::Float(32), "B");

  auto r=te::reduce_axis(Range(0,1),"rv");
  
  te::Tensor C = te::compute(
      {M, N}, std::function<te::TslExpr(tir::Var, tir::Var)>([=](tir::Var i, tir::Var j) {
        return tsl_sum(tir::TslGemm(A.TslPLoad({i,r}),B.TslPLoad({r,j})),{r});


      }),
      "gemm(A,B)");

  auto sch = te::create_schedule({C->op});
  te::TslPrintDecomposeCtx(sch[C].as<te::StageNode>());
  Array<tir::IterVar> ret;
  sch[C].decompose({64, 64}, ret);
  te::TslPrintDecomposeCtx(sch[C].as<te::StageNode>());

  auto req = te::CollectDim(Downcast<tir::TslReduce>(C->op.as<te::ComputeOpNode>()->body[0]));
  for (auto& kv : req) {
    cout << kv.first << ":" << kv.second << endl;
  }

}

