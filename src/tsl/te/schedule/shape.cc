#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tsl/te/expr_shape_infer.h>
namespace tvm {
namespace te {

#define TSL_CHECK_BINOP_CHILDS_VISITED     \
  CHECK(dim_c_ind_map.count(op->a.get())); \
  CHECK(dim_c_ind_map.count(op->b.get()))

#define TSL_CHECK_BINOP_CHILDS_VISITED_AND_NDIM_OF(Size) \
  TSL_CHECK_BINOP_CHILDS_VISITED;                        \
  CHECK(dim_c_ind_map[op->a.get()].size() == Size);      \
  CHECK(dim_c_ind_map[op->b.get()].size() == Size)

class TslExprDimCollectorNChecker final : public ExprVisitor {
 public:
  explicit TslExprDimCollectorNChecker() = default;
  using ExprVisitor::VisitExpr;
  void VisitExpr_(const TslProducerLoadNode* op) override {
    dim_c_ind_map[op] = op->c_indices;
  }
  void VisitExpr_(const TslAddNode* op) override {
    if (!in_combiner) {
      this->VisitExpr(op->a);
      this->VisitExpr(op->b);
      TSL_CHECK_BINOP_CHILDS_VISITED_AND_NDIM_OF(2);
      // TODO: design a checker to confirm that every index primexpr(var) is the same for every dim
      // resp. for tslAdd CHECK(dim_c_ind_map[op->a.get()].same_as(dim_c_ind_map[op->b.get()]));
      dim_c_ind_map[op] = dim_c_ind_map[op->a.get()];
    } else {
      CHECK(prop.defined());
      dim_c_ind_map[op] = prop;
    }
  }
  void VisitExpr_(const TslVarNode* op) override {
    CHECK(in_combiner);
    CHECK(prop.defined());
    dim_c_ind_map[op] = prop;
  }
  void VisitExpr_(const TslReduceNode* op) override {
    CHECK(op->source.size() == 1);
    CHECK(op->combiner->result.size() == 1);
    this->VisitExpr(op->source[0]);
    // source request should be established at this point
    CHECK(dim_c_ind_map.count(op->source[0].get()));
    // prop dim indices to commreducer
    in_combiner = true;
    prop = dim_c_ind_map[op->source[0].get()];
    this->VisitExpr(op->combiner->result[0]);
    in_combiner = false;
    dim_c_ind_map[op] = prop;
  }
  void VisitExpr_(const TslGemmNode* op) override {
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
    TSL_CHECK_BINOP_CHILDS_VISITED_AND_NDIM_OF(2);
    Array<Array<PrimExpr>> out_dim_map;
    switch (op->op_type) {
      case (TslGemmType::kNN): {
        out_dim_map.push_back(dim_c_ind_map[op->a.get()][0]);
        out_dim_map.push_back(dim_c_ind_map[op->b.get()][1]);
        break;
      }
      case (TslGemmType::kNT): {
        out_dim_map.push_back(dim_c_ind_map[op->a.get()][0]);
        out_dim_map.push_back(dim_c_ind_map[op->b.get()][0]);
        break;
      }
      case (TslGemmType::kTN): {
        out_dim_map.push_back(dim_c_ind_map[op->a.get()][1]);
        out_dim_map.push_back(dim_c_ind_map[op->b.get()][1]);
        break;
      }
      case (TslGemmType::kTT): {
        out_dim_map.push_back(dim_c_ind_map[op->a.get()][1]);
        out_dim_map.push_back(dim_c_ind_map[op->b.get()][0]);
        break;
      }
    }
    dim_c_ind_map[op] = out_dim_map;
  }

  void Run(TslExpr e) { this->VisitExpr(e); }
  std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> dim_c_ind_map;

 private:
  bool in_combiner{false};
  Array<Array<PrimExpr>> prop;
};

std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> CollectDim(TslExpr e) {
  TslExprDimCollectorNChecker c;
  c.Run(e);
  return c.dim_c_ind_map;
}

class TslExprShapeAgent final : public ExprVisitor {
 public:
  TslExprShapeAgent(const Stage& stage) : ctx(stage->decompose_ctx) {
    CHECK(stage->op.as<ComputeOpNode>() != nullptr) << "Can only run shape inference on computeop";
    const ComputeOpNode* op = stage->op.as<ComputeOpNode>();
    CHECK(op->attrs.count("TslOp"));
    CHECK(op->body.size() == 1) << "ComputeOp must have body of size 1";
    this->body = Downcast<TslExpr>(op->body[0]);
  }

  void Run() {
    const auto start_shape = ExtractInit();
    out_map[body] = start_shape;
    this->VisitExpr(body);
  }

  void VisitExpr(const PrimExpr& n) final { ExprVisitor::VisitExpr(n); }

  void VisitExpr_(const TslAddNode* op) final {
    const auto self = GetRef<TslAdd>(op);
    CHECK(this->out_map[self].defined());
    const auto self_out_shape = this->out_map[self];
    this->in_map[self][op->a] = self_out_shape;
    this->in_map[self][op->b] = self_out_shape;
    this->out_map[op->a] = self_out_shape;
    this->out_map[op->b] = self_out_shape;
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const TslReduceNode* op) final {
    const auto self = GetRef<TslReduce>(op);
    CHECK(this->out_map[self].defined());
    const auto self_out_shape = this->out_map[self];
  }

  std::unordered_map<TslExpr, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> out_map;
  // inmap[current_tslexpr]->{input_expr:shape}
  std::unordered_map<TslExpr,
                     std::unordered_map<TslExpr, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>,
                     ObjectPtrHash, ObjectPtrEqual>
      in_map;

 private:
  const StageNode::DecomposeContxt& ctx;
  Array<PrimExpr> ExtractInit() const;
  TslExpr body;
};

void InferShape(const Stage);  // TODO

Array<PrimExpr> TslExprShapeAgent::ExtractInit() const {
  Array<PrimExpr> ret;
  CHECK(!this->ctx.empty()) << "dim size has to be greater than 0!";
  const size_t s_size = this->ctx[0].size();
  for (size_t dim = 0; dim < this->ctx.size(); dim++) {
    auto& stack = ctx[dim];
    if (stack.iter_type == IterVarType::kDataPar) {
      CHECK(stack.size() == s_size) << "decompstack having different depth is illegal";
      ret.push_back(stack[s_size - 1].factor);
    }
  }
  return ret;
}

}  // namespace te
}  // namespace tvm