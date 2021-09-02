#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tsl/te/expr_shape_infer.h>
#include <tvm/tsl/tir/tsl_downcast.h>
#include "cind_utility.h"
namespace tvm {
namespace te {

#define TSL_CHECK_BINOP_CHILDS_VISITED     \
  CHECK(dim_c_ind_map.count(op->a.get())); \
  CHECK(dim_c_ind_map.count(op->b.get()))

#define TSL_CHECK_BINOP_CHILDS_VISITED_AND_NDIM_OF(Size) \
  TSL_CHECK_BINOP_CHILDS_VISITED;                        \
  CHECK(dim_c_ind_map[op->a.get()].size() == Size);      \
  CHECK(dim_c_ind_map[op->b.get()].size() == Size)


bool IsPureCIndex(const Array<PrimExpr>& c_index) {
  return c_index.size()==1;
}

bool IsCIndexEqual(const Array<PrimExpr>& ca, const Array<PrimExpr>& cb) {
  const size_t size_a=ca.size();
  const size_t size_b=cb.size();
  if (size_a!=size_b) return false;
  for (size_t i=0;i<size_a;i++) {
    const auto ptr_a = ca[i].get();
    const auto ptr_b = cb[i].get();
    if (ptr_a != ptr_b) return false;
  }
  return true;
}

size_t FindCIndex(const Array<PrimExpr>& c_index, const Array<Array<PrimExpr>>& target_c_indices) {
  for (size_t i=0;i<target_c_indices.size();i++) {
    if (IsCIndexEqual(c_index,target_c_indices[i])) {
      return i;
    }
  }
  return target_c_indices.size();
}


bool IsCIndicesEqual(const Array<Array<PrimExpr>>& ca,const Array<Array<PrimExpr>>& cb) {
  const size_t size_a = ca.size();
  const size_t size_b = cb.size();
  if (size_a != size_b) return false;
  for (size_t i = 0; i < size_a; i++) {
    if (!IsCIndexEqual(ca[i],cb[i])) return false;
  }
  return true;
}


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
      CHECK(IsCIndicesEqual(dim_c_ind_map[op->a.get()],dim_c_ind_map[op->b.get()]));
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
    this->VisitExpr(op->combiner->lhs[0]);
    this->VisitExpr(op->combiner->rhs[0]);
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

  void VisitExpr_(const TslConvNode* op) override {
    this->VisitExpr(op->b); //visit kernel first
    this->VisitExpr(op->a);
    Array<Array<PrimExpr>> out_dim_map;
    switch (op->op_type) {
      case (TslConvType::kNHWC_HWIO): {
        TSL_CHECK_BINOP_CHILDS_VISITED_AND_NDIM_OF(4);
        out_dim_map.push_back(dim_c_ind_map[op->a.get()][0]);
        const auto ih=dim_c_ind_map[op->a.get()][1];
        const auto rh=dim_c_ind_map[op->b.get()][0];
        out_dim_map.push_back(_extract_nonreduction_for_conv(ih,rh));
        const auto iw = dim_c_ind_map[op->a.get()][2];
        const auto rw = dim_c_ind_map[op->b.get()][1];
        out_dim_map.push_back(_extract_nonreduction_for_conv(iw, rw));
        out_dim_map.push_back(dim_c_ind_map[op->b.get()][3]);
      }
    }
    dim_c_ind_map[op]=out_dim_map;
  }

  void Run(const TslExpr& e) {
    this->VisitExpr(e);
  }
  std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> dim_c_ind_map;

 private:
  bool in_combiner{false};
  Array<Array<PrimExpr>> prop;
  Array<PrimExpr> _extract_nonreduction_for_conv(Array<PrimExpr> c_index, Array<PrimExpr> r_c_index) {
    CHECK(r_c_index.size()==1);
    CHECK(c_index.size()==2);
    Array<PrimExpr> ret;
    for (auto &v:c_index) {
      if (v.get()==r_c_index[0].get()) continue;
      ret.push_back(v);
    }
    return ret;
  }
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

    this->body = TslRuntimeDowncast(op->body[0]);
  }
  using ExprVisitor::VisitExpr;

  void Run() {
    dim_c_ind_map=CollectDim(this->body);
    const auto start_shape = ExtractInit();
    out_map[body.get()] = start_shape;
    this->VisitExpr(body);
  }

  void VisitExpr_(const TslProducerLoadNode* op) override {
    CHECK(this->out_map[op].defined());
    return;
  }

  void VisitExpr_(const TslAddNode* op) override {
    CHECK(this->out_map[op].defined());
    const auto self_out_shape = this->out_map[op];
    //this->in_map[self][op->a] = self_out_shape;
    //this->in_map[self][op->b] = self_out_shape;
    this->out_map[op->a.get()] = self_out_shape;
    this->out_map[op->b.get()] = self_out_shape;
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }


/*
  #define TSL_SHAPEAGENT_C_INDEX_SHAPE_COLLECT_FATAL_NONPURE(name) \
  const auto& dim_c_indices_##name =dim_c_ind_map[op->name.get()];        \
  Array<PrimExpr> out_shape_##__COUNTER__;                                       \
  for (auto& c_index : dim_c_indices_##name) {                            \
    if (IsPureCIndex(c_index)) {                                          \
      size_t loc = FindCIndex(c_index, self_dim_c_indices);               \
      if (loc < self_dim_c_indices.size()) {                              \
        out_shape_##name.push_back(this->out_map[op][loc]);               \
      } else {                                                            \
        auto& stack = ctx.find(c_index[0]);                               \
        out_shape_##name.push_back(stack[stack.size() - 1].factor);       \
      }                                                                   \
    } else {                                                              \
      LOG_FATAL;                                                          \
    }                                                                     \
  }                                                                       \
  this->out_map[op->name.get()] = out_shape_##name
*/

  void VisitExpr_(const TslGemmNode* op) override {
    CHECK(this->out_map[op].defined());
    const auto& dim_c_indices_A= dim_c_ind_map[op->a.get()];
    Array<PrimExpr> out_shape_A;
    for (auto& c_index:dim_c_indices_A) {
      if (IsPureCIndex(c_index)) {
        out_shape_A.push_back(GetPureCIndexShape(c_index, op));
      } else { //Not pure -- for now consider this illegal for TslGemmNode
        LOG_FATAL;
      }
    }
    this->out_map[op->a.get()]=out_shape_A;
    const auto& dim_c_indices_B = dim_c_ind_map[op->b.get()];
    Array<PrimExpr> out_shape_B;
    for (auto& c_index : dim_c_indices_B) {
      if (IsPureCIndex(c_index)) {
        out_shape_B.push_back(GetPureCIndexShape(c_index,op));
      } else {  // Not pure -- for now consider this illegal for TslGemmNode
        LOG_FATAL;
      }
    }
    this->out_map[op->b.get()] = out_shape_B;
    this->VisitExpr(op->a);
    this->VisitExpr(op->b);
  }

  void VisitExpr_(const TslConvNode* op) override {
    CHECK(this->out_map[op].defined());
    const auto& dim_c_indices_k=dim_c_ind_map[op->b.get()];
    Array<PrimExpr> out_shape_k;
    for (auto& c_index:dim_c_indices_k) {
      if (IsPureCIndex(c_index)) {
        out_shape_k.push_back(GetPureCIndexShape(c_index,op));
      } else { //consider kernel having non-pure C_index illegal
        LOG_FATAL;
      }
    }
    this->out_map[op->b.get()]=out_shape_k;
    const auto& dim_c_indices_i=dim_c_ind_map[op->a.get()];
    Array<PrimExpr> out_shape_i;
    size_t dim=0;
    for (auto& c_index:dim_c_indices_i) {
      if (IsPureCIndex(c_index)) {
        out_shape_i.push_back(GetPureCIndexShape(c_index,op));
      }else { //okay, here comes the real thing...
        CHECK(c_index.size()==2)<<"only support c_index.size()==2 for now";
        Map<Var,IntSet> dom_map;
        
        for (auto& pure:c_index) {
          CHECK(pure.as<VarNode>());
          PrimExpr shape = GetPureCIndexShape({pure}, op);
          auto interval =IntSet::FromRange(Range(0,shape));
          dom_map.Set(Downcast<Var>(pure), interval);
        }
        std::cout<<op->strides<<std::endl;
        PrimExpr conv_stride=op->strides[dim];
        PrimExpr eval_expr=c_index[0]*conv_stride+c_index[1];
        IntSet eval_set = arith::EvalSet(eval_expr, dom_map);
        CHECK(is_zero(eval_set.min()));
        PrimExpr shape = eval_set.max()+1;
        out_shape_i.push_back(shape);
      }
      dim++;
    }
    this->out_map[op->a.get()]=out_shape_i;
  }

  void VisitExpr_(const TslReduceNode* op) override {
    CHECK(this->out_map[op].defined());
    const auto& dim_c_indices_source= dim_c_ind_map[op->source[0].get()];
    Array<PrimExpr> out_shape_source;
    for (auto& c_index:dim_c_indices_source) {
      if (IsPureCIndex(c_index)) {
        out_shape_source.push_back(GetPureCIndexShape(c_index,op));
      } else { //Not pure -- for now also consider this illegal for TslReduce
        LOG_FATAL;
      }
    }
    this->out_map[op->source[0].get()]=out_shape_source;
    //combiner:
    this->out_map[op->combiner->result[0].get()]=out_shape_source;
    this->out_map[op->combiner->lhs[0].get()]=out_shape_source;
    this->out_map[op->combiner->rhs[0].get()]=out_shape_source;
    this->VisitExpr(op->source[0]);
  }

  std::unordered_map<const TslExprNode*, Array<PrimExpr>> out_map;

  //NOT NECCESARY
  // inmap[current_tslexpr]->{input_expr:shape}
  std::unordered_map<TslExpr,
                     std::unordered_map<TslExpr, Array<PrimExpr>, ObjectPtrHash, ObjectPtrEqual>,
                     ObjectPtrHash, ObjectPtrEqual> in_map;
     

 private:
  PrimExpr GetPureCIndexShape(const Array<PrimExpr>& child_pure, const TslExprNode* op);
  const StageNode::DecomposeContxt& ctx;
  std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> dim_c_ind_map;
  Array<PrimExpr> ExtractInit();
  
  TslExpr body;
};

std::unordered_map<const TslExprNode*, Array<PrimExpr>> InferShape(const Stage& stage) {
  TslExprShapeAgent agent(stage);
  agent.Run();
  return agent.out_map;
}

PrimExpr TslExprShapeAgent::GetPureCIndexShape(const Array<PrimExpr>& child_pure, const TslExprNode* op) {
  const auto& parent_map=this->dim_c_ind_map[op];
  size_t loc = FindCIndex(child_pure, parent_map);
  if (loc < parent_map.size()) {
    return this->out_map[op][loc];
  } else {
    auto& stack = ctx.find(child_pure[0]);
    return stack[stack.size() - 1].factor;
  }
}




Array<PrimExpr> TslExprShapeAgent::ExtractInit() {
  Array<PrimExpr> ret;
  const size_t s_size = this->ctx[0].size();
  auto& c_indices=this->dim_c_ind_map[body.get()];
  for (auto& c_index:c_indices) {
    std::cout<<c_index<<std::endl;
    CHECK(IsPureCIndex(c_index));
    auto stack=ctx.find(c_index[0]);
    CHECK(stack.size() == s_size) << "decompstack having different depth is illegal";
    ret.push_back(stack[s_size-1].factor);
    std::cout<<ret<<std::endl;
  }
  return ret;
}

}  // namespace te
}  // namespace tvm