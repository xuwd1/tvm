#ifndef TVM_TSL_TIR_EXPR_H_
#define TVM_TSL_TIR_EXPR_H_

#include <tvm/ir/expr.h>
#include <tvm/node/container.h>
#include <tvm/node/functor.h>
#include <tvm/node/node.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/var.h>


#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace tir {

#define TVM_TSL_DECLARE_BINOP_CONSTRUCTOR(Name) TVM_DLL Name(TslExpr a, TslExpr b)


class TslExprNode : public PrimExprNode {
 public:
  static constexpr const char* _type_key = "TslExpr";
  static constexpr const uint32_t _type_child_slots = 32;
  TVM_DECLARE_BASE_OBJECT_INFO(TslExprNode, PrimExprNode);
};

class TslExpr : public PrimExpr {
 public:
  virtual Array<Array<PrimExpr>> PropbackElemshape(Array<PrimExpr> source){ 
    CHECK_EQ(0, 1);
    return Array<Array<PrimExpr>>();
  }
  TVM_DEFINE_OBJECT_REF_METHODS(TslExpr, PrimExpr, TslExprNode);
};

template <typename T>
class TslBinaryOpNode : public TslExprNode {
 public:
  /*! \brief The left operand. */
  TslExpr a;
  /*! \brief The right operand. */
  TslExpr b;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("a", &a);
    v->Visit("b", &b);
  }

  bool SEqualReduce(const T* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(a, other->a) && equal(b, other->b);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(a);
    hash_reduce(b);
  }

  TVM_DECLARE_FINAL_OBJECT_INFO(T, PrimExprNode);
};

class TslAddNode : public TslBinaryOpNode<TslAddNode> {
 public:
  static constexpr const char* _type_key = "tir.TslAdd";
};

class TslAdd : public TslExpr {
 public:
  Array<Array<PrimExpr>> PropbackElemshape(Array<PrimExpr> source) final;
  TVM_TSL_DECLARE_BINOP_CONSTRUCTOR(TslAdd);
  TVM_DEFINE_OBJECT_REF_METHODS(TslAdd, TslExpr, TslAddNode);
};

class TslProducerLoadNode : public TslExprNode {
 public:
  /*! \brief The buffer producer. */
  DataProducer producer;
  /*! \brief The location arguments. */
  Array<PrimExpr> indices;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &(this->dtype));
    v->Visit("producer", &producer);
    v->Visit("indices", &indices);
  }

  bool SEqualReduce(const TslProducerLoadNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(producer, other->producer) &&
           equal(indices, other->indices);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(producer);
    hash_reduce(indices);
  }

  static constexpr const char* _type_key = "tir.TslProducerLoad";
  TVM_DECLARE_FINAL_OBJECT_INFO(TslProducerLoadNode, PrimExprNode);
};

/*!
 * \brief Managed reference to ProducerLoadNode.
 * \sa ProducerLoadNode
 */
class TslProducerLoad : public TslExpr {
 public:
  TVM_DLL explicit TslProducerLoad(DataProducer producer, Array<PrimExpr> indices);
  Array<Array<PrimExpr>> PropbackElemshape(Array<PrimExpr> source) final;
  TVM_DEFINE_OBJECT_REF_METHODS(TslProducerLoad, TslExpr, TslProducerLoadNode);
};


}  // namespace tir
}  // namespace tvm

#endif