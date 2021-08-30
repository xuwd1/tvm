#include <tvm/tir/expr.h>
#include <tvm/te/schedule.h>
#include <tvm/te/operation.h>
#include <tvm/tir/expr_functor.h>
namespace tvm {
namespace te {

class TslExprShapeAgent final: public ExprVisitor {
public:
  TslExprShapeAgent(const Stage& stage):ctx(stage->decompose_ctx) {
    CHECK(stage->op.as<ComputeOpNode>()!=nullptr)<<"Can only run shape inference on computeop";
    const ComputeOpNode* op=stage->op.as<ComputeOpNode>();
    CHECK(op->attrs.count("TslOp"));
    CHECK(op->body.size()==1)<<"ComputeOp must have body of size 1";
    this->body = Downcast<TslExpr>(op->body[0]);
  }

  void Run() {
    const auto start_shape=ExtractInit();
    out_map[body]=start_shape;
    this->VisitExpr(body);
  }

  void VisitExpr(const PrimExpr& n) final {
    ExprVisitor::VisitExpr(n);
  }

  void VisitExpr_(const TslAddNode* op) final {
    const auto self=GetRef<TslAdd>(op);
    CHECK(this->out_map[self].defined());
    const auto self_out_shape=this->out_map[self];
    this->in_map[self][op->a] = self_out_shape;
    this->in_map[self][op->b] = self_out_shape;
    this->out_map[op->a]=self_out_shape;
    this->out_map[op->b] = self_out_shape;
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const TslReduceNode* op) final {
    const auto self=GetRef<TslReduce>(op);
    CHECK(this->out_map[self].defined());
    const auto self_out_shape=this->out_map[self];


  }



  std::unordered_map<TslExpr, Array<PrimExpr>> out_map;
  //inmap[current_tslexpr]->{input_expr:shape}
  std::unordered_map<TslExpr,std::unordered_map<TslExpr,Array<PrimExpr>>> in_map;



private:
  const StageNode::DecomposeContxt& ctx;
  Array<PrimExpr> ExtractInit() const;
  TslExpr body;

  
};


void InferShape(const Stage); //TODO

Array<PrimExpr> TslExprShapeAgent::ExtractInit() const {
  Array<PrimExpr> ret;
  CHECK(!this->ctx.empty())<<"dim size has to be greater than 0!";
  const size_t s_size=this->ctx[0].size();
  for (size_t dim=0;dim<this->ctx.size();dim++) {
    auto& stack=ctx[dim];
    if (stack.iter_type==IterVarType::kDataPar) {
      CHECK(stack.size() == s_size) << "decompstack having different depth is illegal";
      ret.push_back(stack[s_size - 1].factor);
    }
  }
  return ret;
}



}
}