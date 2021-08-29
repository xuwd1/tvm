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
  std::unordered_map<TslExprNode*, Array<PrimExpr>> out_map;
  //inmap[current_tslexpr]->{input_expr:shape}
  std::unordered_map<TslExprNode*,std::unordered_map<TslExprNode*,Array<PrimExpr>>> in_map;

  void VisitExpr_(const TslReduceNode* op) final {
    
  }

private:
  const StageNode::DecomposeContxt& ctx;
  bool start_=false;
  TslExpr body;

  
};


void InferShape(const Stage); //TODO
}
}