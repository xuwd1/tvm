#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tsl/tir/expr.h>
#include <tvm/tsl/tir/stmt_functor.h>
#include <limits>
#include <memory>


namespace tvm {
namespace tir {

#define TVM_DEFINE_BINOP_CONSTRUCTOR(Name)                            \
  Name::Name(TslExpr a, TslExpr b) {                                \
    using T = Name::ContainerType;                                    \
    CHECK(a.defined()) << "ValueError: a is undefined\n";             \
    CHECK(b.defined()) << "ValueError: b is undefined\n";             \
    CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n"; \
    ObjectPtr<T> node = make_object<T>();                             \
    node->dtype = a.dtype();                                          \
    node->a = std::move(a);                                           \
    node->b = std::move(b);                                           \
    data_ = std::move(node);                                          \
  }

#define TVM_DEFINE_UOP_CONSTRUCTOR(Name)                              \
  Name::Name(PrimExpr a, Array<PrimExpr> a_shape) {                   \
    using T = Name::ContainerType;                                    \
    CHECK(a.defined()) << "ValueError: a is undefined\n";             \
    CHECK(a_shape.defined()) << "ValueError: a_shape is undefined\n"; \
    ObjectPtr<T> node = make_object<T>();                             \
    node->dtype = a.dtype();                                          \
    node->a = std::move(a);                                           \
    node->a_shape = std::move(a_shape);                               \
    data_ = std::move(node);                                          \
  }


TslIntImm::TslIntImm(DataType dtype, int64_t value) { 
  CHECK(dtype.is_scalar());
  CHECK(dtype.is_int()) << "only support int for now";
  auto n = make_object<TslIntImmNode>();
  n->dtype = dtype;
  n->value = value;
  data_ = std::move(n);
}

// TODO: global registration of TslInt

TVM_REGISTER_NODE_TYPE(TslIntImmNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslIntImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslIntImmNode*>(node.get());
      if (op->dtype == DataType::Int(32)) {
        p->stream << "TslInt(" << op->value << ")";
      } else {
        p->stream << "(" << op->dtype << ")" << op->value;
      }
    });

TslFloatImm::TslFloatImm(DataType dtype, double value) { 
  CHECK(dtype.is_scalar());
  CHECK(dtype.is_float());
  auto n = make_object<TslFloatImmNode>(); 
  n->dtype = dtype;
  n->value = value;
  data_ = std::move(n);
}

// TODO: global registration of TslFloat

TVM_REGISTER_NODE_TYPE(TslFloatImmNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslFloatImmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslFloatImmNode*>(node.get());
      auto& stream = p->stream;
      stream << "TslFloat(";
      switch (op->dtype.bits()) {
        case 64:
          stream << op->value;
          break;
        case 32:
          stream << op->value << 'f';
          break;
        case 16:
          stream << op->value << 'h';
          break;
        default:
          LOG(FATAL) << "Unknown float type bits=" << op->dtype.bits();
      }
      stream << ")";
    });

TslVar::TslVar(String name_hint, DataType dtype) {
   auto n = make_object<TslVarNode>();
   n->name_hint = std::move(name_hint);
   n->dtype = std::move(dtype);
   data_ = std::move(n);
}

//TODO: global registration of TslVar

TVM_REGISTER_NODE_TYPE(TslVarNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslVarNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslVarNode*>(node.get());
      p->stream << op->name_hint;
    });


TslCommReducer::TslCommReducer(Array<TslVar> lhs, Array<TslVar> rhs, Array<TslExpr> result,
                               Array<TslExpr> identity_element) {
  auto n = make_object<TslCommReducerNode>();
  n->lhs = lhs;
  n->rhs = rhs;
  n->result = result;
  n->identity_element = identity_element;
  data_ = std::move(n);
}

Array<TslExpr> TslCommReducerNode::operator()(Array<TslExpr> a, Array<TslExpr> b) const {
  CHECK_EQ(a.size(), b.size());
  CHECK_EQ(lhs.size(), a.size());
  CHECK_EQ(rhs.size(), b.size());
  Map<TslVar,TslExpr> value_map;
  for (size_t i = 0; i < a.size(); ++i) {
    value_map.Set(lhs[i], a[i]);
    value_map.Set(rhs[i], b[i]);
  }
  auto ret = this->result;
  ret.MutateByApply([&value_map](const TslExpr& e) { return Substitute(e, value_map); });
  return ret;
}

//TODO:global registration of TslCommReducer

TVM_REGISTER_NODE_TYPE(TslCommReducerNode);
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslCommReducerNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslCommReducerNode*>(node.get());
      p->stream << "tsl_comm_reducer(result=" << op->result << ", lhs=" << op->lhs
                << ", rhs=" << op->rhs << ", identity_element=" << op->identity_element << ")";
    });


TslReduce::TslReduce(TslCommReducer combiner, Array<TslExpr> src, Array<IterVar> axis,
                     PrimExpr condition, int value_index, Array<TslExpr> init) {
  for (size_t i = 0; i < axis.size(); i++) {
    CHECK_EQ(axis[i]->iter_type, kCommReduce) << "Can only take axis created by reduce_axis";
  }
  if (!condition.defined()) {
    condition = const_true();
  }
  auto n = make_object<TslReduceNode>();
  CHECK(src.defined());
  if (!init.empty()) {
    CHECK_EQ(init.size(), src.size()) << "Number of inits should match number of exprs";
    for (size_t i = 0; i < init.size(); i++) {
      //TODO:check these
      std::cout << __FILE__ << __LINE__ << std::endl;
      CHECK(init[i]->IsInstance<ProducerLoadNode>() || init[i]->IsInstance<IntImmNode>() ||
            init[i]->IsInstance<FloatImmNode>())
          << "init can only be a IntImm, FloatImm or ProducerLoad";
    }
  }
  n->dtype = src[value_index].dtype();
  n->combiner = std::move(combiner);
  n->source = std::move(src);
  n->init = std::move(init);
  n->axis = std::move(axis);
  n->condition = condition;
  n->value_index = value_index;
  data_ = std::move(n);
}

//TODO:global registration of TslReduce

TVM_REGISTER_NODE_TYPE(TslReduceNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslReduceNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslReduceNode*>(node.get());
      p->stream << "tsl_reduce(combiner=" << op->combiner;
      p->stream << ", source=" << op->source;
      p->stream << ", init=" << op->init;
      p->stream << ", axis=" << op->axis;
      p->stream << ", where=" << op->condition;
      p->stream << ", value_index=" << op->value_index;
      p->stream << ")";
    });


// TslAdd
TVM_DEFINE_BINOP_CONSTRUCTOR(TslAdd);

TVM_REGISTER_NODE_TYPE(TslAddNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslAddNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslAddNode*>(node.get());
      p->stream << "TslAdd(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ')';
    });

TVM_REGISTER_GLOBAL("tir.TslAdd").set_body_typed([](TslExpr a, TslExpr b) {
  return TslAdd(a, b);
});

//TslGemm

TslGemm::TslGemm(TslExpr a, TslExpr b, TslGemmType type) {
  using T=TslGemm::ContainerType;
  CHECK(a.defined()) << "ValueError: a is undefined\n";
  CHECK(b.defined()) << "ValueError: b is undefined\n";
  CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n";
  ObjectPtr<T> node = make_object<T>();
  node->dtype = a.dtype();
  node->a = std::move(a);
  node->b = std::move(b);
  node->op_type = type;
  data_ = std::move(node);
}


TVM_REGISTER_NODE_TYPE(TslGemmNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslGemmNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslGemmNode*>(node.get());
      p->stream << "TslGemm<<"<<op->op_type<<"(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ')';
    });

TVM_REGISTER_GLOBAL("tir.TslGemm").set_body_typed([](TslExpr a, TslExpr b) { return TslGemm(a, b); });

TslConv::TslConv(TslExpr a, TslExpr b, TslConvType type) {
  using T=TslConv::ContainerType;
  CHECK(a.defined()) << "ValueError: a is undefined\n";
  CHECK(b.defined()) << "ValueError: b is undefined\n";
  CHECK(a.dtype() == b.dtype()) << "TypeError: mismatched types\n";
  ObjectPtr<T> node = make_object<T>();
  node->dtype = a.dtype();
  node->a = std::move(a);
  node->b = std::move(b);
  node->op_type = type;
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(TslConvNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslConvNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslConvNode*>(node.get());
      p->stream << "TslConv<<" << op->op_type << "(";
      p->Print(op->a);
      p->stream << ", ";
      p->Print(op->b);
      p->stream << ')';
    });

TVM_REGISTER_GLOBAL("tir.TslConv").set_body_typed([](TslExpr a, TslExpr b) {
  return TslGemm(a, b);
});

// ProducerLoad
TslProducerLoad::TslProducerLoad(DataProducer producer, Array<PrimExpr> indices) {
  ObjectPtr<TslProducerLoadNode> node = make_object<TslProducerLoadNode>();
  node->dtype = producer->GetDataType();
  node->producer = std::move(producer);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

tvm::tir::TslProducerLoad::TslProducerLoad(DataProducer producer,
                                           Array<Array<PrimExpr>> c_indices) {
  auto node=make_object<TslProducerLoadNode>();
  node->dtype=producer->GetDataType();
  node->producer = std::move(producer);
  node->c_indices=std::move(c_indices);
  data_=std::move(node);
}



TVM_REGISTER_GLOBAL("tir.TslProducerLoad")
    .set_body_typed([](DataProducer producer, Array<PrimExpr> indices) {
      return TslProducerLoad(producer, indices);
    });

TVM_REGISTER_NODE_TYPE(TslProducerLoadNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TslProducerLoadNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TslProducerLoadNode*>(node.get());
      p->stream << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) {
          p->stream << ", ";
        }
      }
      p->stream << "]";
    });



}
}  // namespace tvm