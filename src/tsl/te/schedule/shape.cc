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

  void Run(TslExpr e) { this->VisitExpr(e); }
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
    this->body = Downcast<TslExpr>(op->body[0]);
  }
  using ExprVisitor::VisitExpr;

  void Run() {
    dim_c_ind_map=CollectDim(this->body);
    //const auto start_shape = ExtractInit();
    //out_map[body] = start_shape;
    this->VisitExpr(body);
  }


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
                     ObjectPtrHash, ObjectPtrEqual> in_map;
     

 private:
  const StageNode::DecomposeContxt& ctx;
  std::unordered_map<const TslExprNode*, Array<Array<PrimExpr>>> dim_c_ind_map;
  std::vector<std::pair<Array<PrimExpr>,PrimExpr>> c_index_shape_lut;
  Array<PrimExpr> ExtractInit() const;
  Array<PrimExpr> GetOutShapeFor(TslExprNode* op);
  PrimExpr AllocShapeForCIndex(Array<PrimExpr> c_index);
  bool _lutlookup(const Array<PrimExpr>& c_index, PrimExpr& ret);
  TslExpr body;
};

void InferShape(const Stage);  // TODO

Array<PrimExpr> TslExprShapeAgent::GetOutShapeFor(TslExprNode* op) {
  const auto dim_c_indices_map=this->dim_c_ind_map[op];
  Array<PrimExpr> ret;
  for (auto& c_index:dim_c_indices_map) {
    PrimExpr shape;
    if (_lutlookup(c_index, shape)) {  // hit
      ret.push_back(shape);
    } else {
      
    }
  }
  return ret;

}

PrimExpr TslExprShapeAgent::AllocShapeForCIndex(Array<PrimExpr> c_index) {
  return PrimExpr();


}

bool TslExprShapeAgent::_lutlookup(const Array<PrimExpr>& c_index, PrimExpr& ret) {
  for (const auto& pair:this->c_index_shape_lut) {
    const auto& lut_c_index=pair.first;
    if (IsCIndexEqual(lut_c_index,c_index)) {
      ret=pair.second;
      return true;
    }
  }
  return false;
}


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