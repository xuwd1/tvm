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

  const int wmma_M = 16;
  const int wmma_N = 16;
  const int wmma_K = 16;
  const int warp_M_tiles = 2;
  const int warp_N_tiles = 2;
  const int block_M_warps = 4;
  const int block_N_warps = 4;
  

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

  auto AS = sch.cache_read(A, "shared", {C->op});
  auto BS = sch.cache_read(B, "shared", {C->op});
  auto AL = sch.cache_read(AS, "wmma.matrix_a", {C->op});
  auto BL = sch.cache_read(BS, "wmma.matrix_b", {C->op});
  auto CL = sch.cache_write(C, "wmma.accumulator");
  auto CS = sch.cache_read(CL, "shared", {C->op});



  Array<tir::IterVar> block_ivar;
  sch[C].decompose({wmma_M * warp_M_tiles * block_M_warps, wmma_N * warp_N_tiles * block_N_warps},
                   block_ivar);
  printDecompDBGInfo(sch[C].operator->());
  
  const int vec = 8;
  const int warp_size = 32;
  Array<tir::IterVar> thread_ivars;
  sch[C].decompose({1, vec}, thread_ivars);
  tir::IterVar threads;
  sch[C].fuse(thread_ivars, &threads);
  tir::IterVar t1, tid_x;
  tir::IterVar t2, tid_y;
  tir::IterVar t3, tid_z;
  sch[C].split(threads, warp_size, &t1 ,& tid_x);
  sch[C].split(t1, block_M_warps, &t2, &tid_y);
  sch[C].split(t2, block_N_warps, &t3, &tid_z);

  sch[CS].compute_at(sch[C], block_ivar[1]);
  Array<tir::IterVar> warps;
  sch[CS].decompose({block_M_warps * wmma_M, block_N_warps * wmma_N}, warps);
  Array<tir::IterVar> warp_tiles;
  sch[CS].decompose({wmma_M, wmma_N}, warp_tiles);

  const int chunk = 4;
  sch[CL].compute_at(sch[CS], warps[1]);
  Array<tir::IterVar> compute_warp_tiles;
  sch[CL].decompose({wmma_M, wmma_N}, compute_warp_tiles);
  Array<tir::IterVar> chunk_ivar;
  //sch[CL].decompose_reduce({wmma_K * chunk}, chunk_ivar);
  Array<tir::IterVar> reduce_tile;
  //sch[CL].decompose_reduce({wmma_K}, reduce_tile);
  sch[CL].reorder({chunk_ivar[0], reduce_tile[0], compute_warp_tiles[0], compute_warp_tiles[1]});
  
  sch[AL].compute_at(sch[CL], reduce_tile[0]);
  sch[AL].decompose({wmma_M, wmma_K}, Array<tir::IterVar>());
  sch[BL].compute_at(sch[CL], reduce_tile[0]);
  sch[BL].decompose({wmma_K, wmma_N}, Array<tir::IterVar>());

  sch[AS].compute_at(sch[CL], chunk_ivar[0]);
  sch[AS].decompose({1, vec}, thread_ivars);
  sch[AS].fuse(thread_ivars, &threads);
  sch[AS].split(threads, warp_size, &t1, &tid_x);
  sch[AS].split(t1, block_M_warps, &t2, &tid_y);
  sch[AS].split(t2, block_N_warps, &t3, &tid_z);
  //sch[AS].bind...

  sch[BS].compute_at(sch[CL], chunk_ivar[0]);
  sch[BS].decompose({1, vec}, thread_ivars);
  sch[BS].fuse(thread_ivars, &threads);
  sch[BS].split(threads, warp_size, &t1, &tid_x);
  sch[BS].split(t1, block_M_warps, &t2, &tid_y);
  sch[BS].split(t2, block_N_warps, &t3, &tid_z);
  

  

  

  // 
}