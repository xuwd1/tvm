/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tensor.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>
#include <tvm/te/tensor_intrin.h>

#include <memory>

namespace tvm {
namespace te {

IterVar thread_axis(Range dom, std::string tag) {
  return IterVar(dom, Var(tag), kThreadIndex, tag);
}

IterVar reduce_axis(Range dom, std::string name) { return IterVar(dom, Var(name), kCommReduce); }

Var var(std::string name_hint, DataType t) { return Var(name_hint, t); }

// Tensor
PrimExpr Tensor::operator()(Array<Var> indices) const {
  Array<PrimExpr> arr(indices.begin(), indices.end());
  return operator()(arr);
}

TslExpr Tensor::TslPload(Array<Array<PrimExpr>> c_indices) const {
  CHECK((*this)->op->attrs.count("TslOp"))<<"compound indexing only available to TslOp";
  if (ndim()!=0) {
    CHECK_EQ(ndim(), c_indices.size()) << "Tensor dimension mismatch in compound indexer"
                                     << "ndim = " << ndim() << ", c_indices.size=" << c_indices.size();
  }
  return TslProducerLoad((*this),c_indices);
}

TslExpr Tensor::operator()(Array<Array<Var>> c_indices) const {
  Array<Array<PrimExpr>> expr_c_indices;
  for (auto &arr:c_indices) {
    expr_c_indices.push_back(Array<PrimExpr>(arr.begin(),arr.end()));
  }
  return TslPload(expr_c_indices);
}

TslExpr Tensor::TslPLoad(Array<PrimExpr> indices) const { 
  if (ndim() != 0) {
    CHECK_EQ(ndim(), indices.size()) << "Tensor dimension mismatch in read"
                                     << "ndim = " << ndim() << ", indices.size=" << indices.size();
  }
  Array<Array<PrimExpr>> expr_c_indices;
  for (auto &v:indices) {
    expr_c_indices.push_back(Array<PrimExpr>({v}));
  }
  return TslProducerLoad((*this), expr_c_indices);
}

PrimExpr Tensor::operator()(Array<PrimExpr> indices) const {
  if (ndim() != 0) {
    CHECK_EQ(ndim(), indices.size()) << "Tensor dimension mismatch in read"
                                     << "ndim = " << ndim() << ", indices.size=" << indices.size();
  }

  return ProducerLoad((*this), indices);
}

String TensorNode::GetNameHint() const {
  return op->num_outputs() == 1 ? op->name : (op->name + ".v" + std::to_string(value_index));
}

Tensor Operation::output(size_t i) const {
  auto node = make_object<TensorNode>();
  node->op = *this;
  node->value_index = i;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  if (this->operator->()->attrs.count("TslOp") != 0) {
    node->write_ushape = (*this)->output_unionshape(i);
    node->write_eshape = (*this)->output_elemshape(i);
    node->read_ushape = Array<PrimExpr>(node->shape.size(), 1);
    node->read_eshape = (*this)->output_shape(i);
  }
  return Tensor(node);
}

Tensor::Tensor(Array<PrimExpr> shape, DataType dtype, Operation op, int value_index) {
  auto n = make_object<TensorNode>();
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->op = op;
  n->value_index = value_index;
  // TODO:investigate if this has anything to do with TSL
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("te.Tensor")
    .set_body_typed([](Array<PrimExpr> shape, DataType dtype, Operation op, int value_index) {
      return Tensor(shape, dtype, op, value_index);
    });

TVM_REGISTER_NODE_TYPE(TensorNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* t = static_cast<const TensorNode*>(node.get());
      p->stream << "Tensor(shape=" << t->shape <<"(read:"<<t->read_ushape<<"<<"<<t->read_eshape<<">>"<<" write:"<<t->write_ushape<<"<<"<<t->write_eshape<<">>)"<< ", op.name=" << t->op->name << ')';
    });

// TensorIntrin
TensorIntrin::TensorIntrin(std::string name, Operation op, Array<Tensor> inputs,
                           Array<Buffer> buffers, Array<Var> scalar_params, Stmt body,
                           Stmt reduce_init, Stmt reduce_update) {
  auto n = make_object<TensorIntrinNode>();
  n->name = std::move(name);
  n->op = std::move(op);
  n->inputs = std::move(inputs);
  n->buffers = std::move(buffers);
  n->scalar_params = std::move(scalar_params);
  n->body = std::move(body);
  n->reduce_init = std::move(reduce_init);
  n->reduce_update = std::move(reduce_update);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("te.TensorIntrin")
    .set_body_typed([](std::string name, Operation op, Array<Tensor> inputs, Array<Buffer> buffers,
                       Array<Var> scalar_params, Stmt body, Stmt reduce_init, Stmt reduce_update) {
      return TensorIntrin(name, op, inputs, buffers, scalar_params, body, reduce_init,
                          reduce_update);
    });

TVM_REGISTER_NODE_TYPE(TensorIntrinNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorIntrinNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TensorIntrinNode*>(node.get());
      p->stream << "TensorIntrin(name=" << op->name << ", " << op << ")";
    });

// TensorIntrinCall
TensorIntrinCall::TensorIntrinCall(TensorIntrin intrin, Array<Tensor> tensors,
                                   Array<Region> regions, Array<IterVar> reduce_axis,
                                   Array<PrimExpr> scalar_inputs) {
  auto n = make_object<TensorIntrinCallNode>();
  n->intrin = std::move(intrin);
  n->tensors = std::move(tensors);
  n->regions = std::move(regions);
  n->reduce_axis = std::move(reduce_axis);
  n->scalar_inputs = std::move(scalar_inputs);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("te.TensorIntrinCall")
    .set_body_typed([](TensorIntrin intrin, Array<Tensor> tensors, Array<Region> regions,
                       Array<IterVar> reduce_axis, Array<PrimExpr> scalar_inputs) {
      return TensorIntrinCall(intrin, tensors, regions, reduce_axis, scalar_inputs);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TensorIntrinCallNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* n = static_cast<const TensorIntrinCallNode*>(node.get());
      p->stream << "TensorIntrinCall(intrin=" << n->intrin << ", " << n << ")";
    });

TVM_REGISTER_NODE_TYPE(TensorIntrinCallNode);

// Other tensor ops.
TVM_REGISTER_GLOBAL("te.TensorEqual").set_body_method(&Tensor::operator==);

TVM_REGISTER_GLOBAL("te.TensorHash").set_body_typed([](Tensor tensor) -> int64_t {
  return static_cast<int64_t>(std::hash<Tensor>()(tensor));
});

TVM_REGISTER_GLOBAL("te.OpGetOutput").set_body_typed([](Operation op, int64_t output) {
  return op.output(static_cast<size_t>(output));
});

TVM_REGISTER_GLOBAL("te.OpNumOutputs").set_body_method<Operation>(&OperationNode::num_outputs);

TVM_REGISTER_GLOBAL("te.OpInputTensors").set_body_method<Operation>(&OperationNode::InputTensors);

}  // namespace te
}  // namespace tvm
