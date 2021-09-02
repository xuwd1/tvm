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
  
  TVM_DEFINE_OBJECT_REF_METHODS(TslExpr, PrimExpr, TslExprNode);
};

class TslIntImmNode : public TslExprNode {
 public:
  int64_t value;
  //TODO: maybe adding shape attrs on tslexprnodes whose values are delay-inferred? 
  void VisitAttrs(AttrVisitor* v) { 
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }
  bool SEqualReduce(const TslIntImmNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }
  static constexpr const char* _type_key = "TslIntImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(TslIntImmNode, TslExprNode);
};

class TslIntImm : public TslExpr {
 public:
  /*!
   * \brief Constructor.
   * \param dtype The data type of the value.
   * \param value The internal value.
   */
  TVM_DLL TslIntImm(DataType dtype, int64_t value);

  TVM_DEFINE_OBJECT_REF_METHODS(TslIntImm, TslExpr, TslIntImmNode);
};

class TslFloatImmNode : public TslExprNode {
 public:
  double value;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("value", &value);
  }

  bool SEqualReduce(const TslFloatImmNode* other, SEqualReducer equal) const {
    return equal(dtype, other->dtype) && equal(value, other->value);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(value);
  }

  static constexpr const char* _type_key = "TslFloatImm";
  TVM_DECLARE_FINAL_OBJECT_INFO(TslFloatImmNode, TslExprNode);
};

class TslFloatImm : public TslExpr {
 public:
  TVM_DLL TslFloatImm(DataType dtype, double value);
  TVM_DEFINE_OBJECT_REF_METHODS(TslFloatImm, TslExpr, TslFloatImmNode);
};


// TODO: NOT equivalent of TVM Var, currently solely for acting as a placeholder in Tslcommreducer
class TslVarNode : public TslExprNode {
 public:
  /*!
   * \brief The hint to the variable name.
   * \note Each variable is uniquely identified by its address.
   */
  String name_hint;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("name", &name_hint);
  }

  bool SEqualReduce(const TslVarNode* other, SEqualReducer equal) const {
    if (!equal(dtype, other->dtype)) return false;
    return equal.FreeVarEqualImpl(this, other);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce.FreeVarHashImpl(this);
  }

  static constexpr const char* _type_key = "tir.TslVar";
  static constexpr const uint32_t _type_child_slots = 1;
  TVM_DECLARE_BASE_OBJECT_INFO(TslVarNode, TslExprNode);
};

// TODO: NOT equivalent of TVM Var, currently solely for acting as a placeholder in Tslcommreducer
class TslVar : public TslExpr {
 public:
  explicit TslVar(ObjectPtr<Object> n) : TslExpr(n) {}
  /*!
   * \brief Constructor
   * \param name_hint variable name
   * \param dtype data type
   */
  TVM_DLL explicit TslVar(String name_hint = "v", DataType dtype = DataType::Int(32));

  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const TslVarNode* operator->() const { return get(); }
  /*!
   * \brief Get pointer to the internal value.
   * \return the corresponding Variable.
   */
  const TslVarNode* get() const { return static_cast<const TslVarNode*>(data_.get()); }
  /*! \brief type indicate the container type */
  using ContainerType = TslVarNode;
};

class TslCommReducerNode : public Object {
 public:
  /*! \brief The left argument of reducer */
  Array<TslVar> lhs;
  /*! \brief The right argument of reducer */
  Array<TslVar> rhs;
  /*! \brief The result of reducer */
  Array<TslExpr> result;
  /*!
   * \brief The identity element of reducer, which leaves other
   *  elements unchanged when combined with it, with respect to
   *  the binary operation of this reducer uses.
   */
  Array<TslExpr> identity_element;
  /*! \brief Function call operator to combine a and b */
  Array<TslExpr> operator()(Array<TslExpr> a, Array<TslExpr> b) const;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("lhs", &lhs);
    v->Visit("rhs", &rhs);
    v->Visit("result", &result);
    v->Visit("identity_element", &identity_element);
  }

  bool SEqualReduce(const TslCommReducerNode* other, SEqualReducer equal) const {
    return equal.DefEqual(lhs, other->lhs) && equal.DefEqual(rhs, other->rhs) &&
           equal(result, other->result) && equal(identity_element, other->identity_element);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce.DefHash(lhs);
    hash_reduce.DefHash(rhs);
    hash_reduce(result);
    hash_reduce(identity_element);
  }

  static constexpr const char* _type_key = "tir.TslCommReducer";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(TslCommReducerNode, Object);
};

class TslCommReducer : public ObjectRef {
 public:
  TVM_DLL TslCommReducer(Array<TslVar> lhs, Array<TslVar> rhs, Array<TslExpr> result,
                         Array<TslExpr> identity_element);

  TVM_DEFINE_OBJECT_REF_METHODS(TslCommReducer, ObjectRef, TslCommReducerNode);
};

class TslReduceNode : public TslExprNode {
 public:
  TslCommReducer combiner;
  Array<TslExpr> source;
  Array<TslExpr> init;
  Array<IterVar> axis;
  PrimExpr condition;
  int value_index;
  void VisitAttrs(AttrVisitor* v) {
    v->Visit("dtype", &dtype);
    v->Visit("combiner", &combiner);
    v->Visit("source", &source);
    v->Visit("init", &init);
    v->Visit("axis", &axis);
    v->Visit("condition", &condition);
    v->Visit("value_index", &value_index);
  }
  bool SEqualReduce(const TslReduceNode* other, SEqualReducer equal) const {
    // check axis first so IterVars can define the necessary variables.
    return equal(dtype, other->dtype) && equal(axis, other->axis) &&
           equal(combiner, other->combiner) && equal(source, other->source) &&
           equal(init, other->init) && equal(condition, other->condition) &&
           equal(value_index, other->value_index);
  }
  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dtype);
    hash_reduce(axis);
    hash_reduce(combiner);
    hash_reduce(source);
    hash_reduce(init);
    hash_reduce(condition);
    hash_reduce(value_index);
  }
  static constexpr const char* _type_key = "tir.TslReduce";
  TVM_DECLARE_FINAL_OBJECT_INFO(TslReduceNode, TslExprNode);
};

class TslReduce : public TslExpr {
 public:
  TVM_DLL TslReduce(TslCommReducer combiner, Array<TslExpr> src, Array<IterVar> rdom,
                    PrimExpr condition, int value_index, Array<TslExpr> init);

  TVM_DEFINE_OBJECT_REF_METHODS(TslReduce, TslExpr, TslReduceNode);
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
  
  TVM_TSL_DECLARE_BINOP_CONSTRUCTOR(TslAdd);
  TVM_DEFINE_OBJECT_REF_METHODS(TslAdd, TslExpr, TslAddNode);
};


enum TslGemmType : int {

  // A[i,r]@B[r,j]->C[i,j]
  kNN = 0,
  // A[i,r]@B[j,r]->C[i,j]
  kNT = 1,
  // A[r,i]@B[r,j]->C[i,j]
  kTN = 2,
  // A[r,i]@B[j,r]->C[i,j]
  kTT = 3
  
};


class TslGemmNode : public TslBinaryOpNode<TslGemmNode> {
 public:
  TslGemmType op_type;
  static constexpr const char* _type_key = "tir.TslGemm";
};

class TslGemm : public TslExpr {
 public:
  TVM_DLL TslGemm(TslExpr a,TslExpr b,TslGemmType type=TslGemmType::kNN);
  TVM_DEFINE_OBJECT_REF_METHODS(TslGemm, TslExpr, TslGemmNode);
};

enum TslConvType : int{
  // A[n,h,w,c] * K[h,w,i,o]->C[n,h,w,c]
  kNHWC_HWIO=0
};


class TslConvNode: public TslBinaryOpNode<TslConvNode> {
public:
  TslConvType op_type;
  Array<PrimExpr> strides;
  static constexpr const char* _type_key="tir.TslConv";
};


class TslConv : public TslExpr {
public:
  TVM_DLL TslConv(TslExpr a,TslExpr b,TslConvType type=TslConvType::kNHWC_HWIO,Array<PrimExpr> strides={-1,1,1,-1});
  TVM_DEFINE_OBJECT_REF_METHODS(TslConv,TslExpr,TslConvNode);
};


class TslProducerLoadNode : public TslExprNode {
 public:
  /*! \brief The buffer producer. */
  DataProducer producer;
  /*! \brief The location arguments. */
  Array<PrimExpr> indices;
  /*! \brief The compound indexer. this is only meaningful in TSL before scheduleOps  */
  Array<Array<PrimExpr>> c_indices;

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
  TVM_DLL explicit TslProducerLoad(DataProducer producer, Array<Array<PrimExpr>> c_indices);
  
  TVM_DEFINE_OBJECT_REF_METHODS(TslProducerLoad, TslExpr, TslProducerLoadNode);
};

}  // namespace tir
}  // namespace tvm

#endif