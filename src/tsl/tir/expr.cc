#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tsl/tir/expr.h>

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

Array<Array<PrimExpr>> TslAdd::PropbackElemshape(Array<PrimExpr> source) {
  Array<Array<PrimExpr>> ret;
  ret.push_back(source);
  ret.push_back(source);
  return ret;
}
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

// ProducerLoad
TslProducerLoad::TslProducerLoad(DataProducer producer, Array<PrimExpr> indices) {
  ObjectPtr<TslProducerLoadNode> node = make_object<TslProducerLoadNode>();
  node->dtype = producer->GetDataType();
  node->producer = std::move(producer);
  node->indices = std::move(indices);
  data_ = std::move(node);
}

Array<Array<PrimExpr>> TslProducerLoad::PropbackElemshape(Array<PrimExpr> source) {
  return Array<Array<PrimExpr>>();
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