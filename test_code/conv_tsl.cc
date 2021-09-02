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
#include <tvm/tsl/te/expr_shape_infer.h>
#include <tvm/tsl/tir/op.h>
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
  const int N = 512;
  const int H = 226;
  const int W = 226;
  const int C = 256;
  const int RH =3;
  const int RW =3;
  const int OC = 64;


  te::Tensor A = te::Tslplaceholder({N, H, W, C}, DataType::Float(32), "A");
  te::Tensor B = te::Tslplaceholder({RH, RW, C, OC}, DataType::Float(32), "K");

  auto rx = te::reduce_axis(Range(0, 1), "rx");
  auto ry = te::reduce_axis(Range(0,1),"ry");
  auto rc = te::reduce_axis(Range(0,1),"rc");
  

  te::Tensor CT = te::Tslcompute(
      {N,H-2,W-2,OC }, std::function<te::TslExpr(tir::Var, tir::Var, tir::Var, tir::Var)>(
        [=](tir::Var n, tir::Var h, tir::Var w, tir::Var c)->tir::TslExpr {
          auto pa= A.TslPLoad({{n}, {h, rx}, {w, ry}, {rc}});
                           const auto pb = B.TslPLoad({Array<PrimExpr>({rx}), {ry}, {rc}, {c}});
          return tsl_sum(tir::TslConv(pa,pb),{rx,ry,rc});
                } 
        ),
      "Conv(A,B)");

  auto sch = te::create_schedule({CT->op});
  te::TslPrintDecomposeCtx(sch[CT].as<te::StageNode>());
  Array<tir::IterVar> ret;
  sch[CT].decompose({N, 1,1,C}, ret); //TODO:CHECK DECOOMPOSE FACTOR NUMBER AND VALUE!
  te::TslPrintDecomposeCtx(sch[CT].as<te::StageNode>());

  sch[CT].decompose_reduction({1,1,C},ret);
  te::TslPrintDecomposeCtx(sch[CT].as<te::StageNode>());
  auto req = te::CollectDim(Downcast<tir::TslReduce>(CT->op.as<te::ComputeOpNode>()->body[0]));
  for (auto& kv : req) {
    cout << kv.first->GetTypeKey() << ":" << kv.second << endl;
  }
  auto map = te::InferShape(sch[CT]);
  for (auto& kv : map) {
    cout << kv.first->GetTypeKey() << ":" << kv.second << endl;
  }
}
