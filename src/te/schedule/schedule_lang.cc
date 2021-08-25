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
 * \file schedule_lang.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/te/operation.h>
#include <tvm/te/schedule.h>
#include <tvm/tsl/tsl_debug.h>
#include <stack>
#include <unordered_set>

#include "graph.h"


namespace tvm {
namespace te {

// find first occurance location in leaf
template <typename T>
size_t FindNodeRef(ArrayNode* array_node, const T& v) {
  const Object* n = v.get();
  for (size_t i = 0; i < array_node->size(); ++i) {
    if (array_node->at(i).get() == n) return i;
  }
  return array_node->size();
}

size_t FindLeafVar(ArrayNode* all_vars, ArrayNode* leaf_vars, const IterVar& v) {
  size_t pos = FindNodeRef(leaf_vars, v);
  if (pos < leaf_vars->size()) return pos;

  if (FindNodeRef(all_vars, v) < all_vars->size()) {
    LOG(FATAL) << "Operate on iter var " << v << "that has already been split";
  } else {
    LOG(FATAL) << "Operate on iter var " << v << "that is not part of the schedule";
  }
  return 0;
}

DataType MatchDataType(std::vector<DataType> dtypes) {
  int max_bits = -1;
  for (const auto& dtype : dtypes) {
    CHECK(dtype.is_int());
    CHECK(dtype.is_scalar());
    max_bits = std::max(max_bits, dtype.bits());
  }
  return DataType::Int(max_bits);
}

void SplitHelper(StageNode* self, IterVar parent, PrimExpr factor, PrimExpr nparts,
                 IterVar* p_outer, IterVar* p_inner) {
  // Check if split is valid.
  CHECK(parent->iter_type == kDataPar || parent->iter_type == kCommReduce ||
        parent->iter_type == kOrdered)
      << "Cannot split on " << IterVarType2String(parent->iter_type);
  IterVar outer = IterVar(Range(), parent->var.copy_with_suffix(".outer"), parent->iter_type);
  IterVar inner = IterVar(Range(), parent->var.copy_with_suffix(".inner"), parent->iter_type);
  *p_outer = outer;
  *p_inner = inner;
  // The splits
  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  size_t pos = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), parent);
  self->relations.push_back(Split(parent, outer, inner, factor, nparts));
  // add vars to all vars
  all_vars.push_back(outer);
  all_vars.push_back(inner);
  // replace the position.
  leaf_vars.erase(leaf_vars.begin() + pos);
  leaf_vars.insert(leaf_vars.begin() + pos, inner);
  leaf_vars.insert(leaf_vars.begin() + pos, outer);
}

Stage::Stage(Operation op) {
  auto n = make_object<StageNode>();
  n->op = op;
  n->origin_op = op;
  n->all_iter_vars = op->root_iter_vars();
  // remove opaque var from leaf.
  Array<IterVar> clean;
  for (IterVar iv : n->all_iter_vars) {
    if (iv->iter_type != kOpaque) clean.push_back(iv);
  }
  if (clean.size() == n->all_iter_vars.size()) {
    n->leaf_iter_vars = n->all_iter_vars;
  } else {
    n->leaf_iter_vars = clean;
  }
  data_ = std::move(n);
}

#if TSL_DBG_V0
Stage::Stage(Operation op, ScheduleNode* schedptr) {
  // TODO: NO IDEA WETHER THIS WOULD WORK FOR SCANOP!!!!
  auto n = make_object<StageNode>();
  n->op = op;
  n->origin_op = op;
  n->parent_sched = schedptr;
  auto root_iter_vars = op->root_iter_vars();  // axis first, then reduce axis
  auto entry = StageNode::DecompEntry();
  entry.factors = op->output_elemshape(0);
  entry.level = 0;
  for (auto v : root_iter_vars) {
    n->all_iter_vars.push_back(v);
    if (v->iter_type != kOpaque) {
      std::ostringstream os;
      os << "." << n->decomp_stack.size();
      auto prefix = os.str();
      IterVar left = IterVar(Range(), v->var.copy_with_suffix(prefix + "L"), v->iter_type);
      IterVar right = IterVar(Range(), v->var.copy_with_suffix(prefix + "R"), v->iter_type);
      n->all_iter_vars.push_back(left);
      n->all_iter_vars.push_back(right);
      n->leaf_iter_vars.push_back(left);
      n->leaf_iter_vars.push_back(right);
      auto split = Split(v, right, left, PrimExpr(), 1);
      n->relations.push_back(split);

      entry.left_ivars.push_back(left);
      entry.right_ivars.push_back(right);
      entry.split_relations.push_back(split);
    }
  }
  n->decomp_stack.push_back(entry);
  data_ = std::move(n);
}
#endif

#if TSL_DBG_V1
Stage::Stage(Operation op, ScheduleNode* schedptr) {
  // TODO: NO IDEA WETHER THIS WOULD WORK FOR SCANOP!!!!
  auto n = make_object<StageNode>();
  n->op = op;
  n->origin_op = op;
  n->parent_sched = schedptr;
  auto root_iter_vars = op->root_iter_vars();  // axis first, then reduce axis
  auto entry = StageNode::DecompEntry();
  entry.factors = op->output_elemshape(0);
  entry.level = 0;
  for (auto v : root_iter_vars) {
    n->all_iter_vars.push_back(v);
    if (v->iter_type != kOpaque) {
      std::ostringstream os;
      os << "." << n->decomp_stack.size();
      auto prefix = os.str();
      IterVar left = IterVar(Range(), v->var.copy_with_suffix(prefix + "L"), v->iter_type);
      IterVar right = IterVar(Range(), v->var.copy_with_suffix(prefix + "R"), v->iter_type);
      n->all_iter_vars.push_back(left);
      n->all_iter_vars.push_back(right);
      n->leaf_iter_vars.push_back(left);
      n->leaf_iter_vars.push_back(right);
      auto split = Split(v, right, left, PrimExpr(), 1);
      n->relations.push_back(split);

      entry.left_ivars.push_back(left);
      entry.right_ivars.push_back(right);
      entry.split_relations.push_back(split);
    }
  }
  n->decomp_stack.push_back(entry);
  data_ = std::move(n);
}
#endif

bool Stage::is_scheduled() const {
  const StageNode* n = operator->();
  return !(n->relations.empty() && n->attach_type == kGroupRoot &&
           n->all_iter_vars.same_as(n->leaf_iter_vars));
}

Stage Stage::GetAttachSpec() const {
  Stage attach_spec = *this;
  while (attach_spec->attach_type == kGroupRoot && attach_spec->group.defined()) {
    attach_spec = attach_spec->group;
  }
  return attach_spec;
}

Stage& Stage::set_scope(std::string scope) {  // NOLINT(*)
  (*this)->scope = scope;
  return *this;
}

Stage& Stage::compute_at(Stage parent, IterVar scope) {  // NOLINT(*)
  CHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  // Group constraint checking.
  Stage group = (*this)->group;
  if (group.defined()) {
    Stage pg = parent->group;
    while (pg.defined() && !pg.same_as(group)) {
      pg = pg->group;
    }
    CHECK(pg.same_as(group)) << "Can only assign compute_at to stages within the same group";
  }

  (*this)->attach_type = kScope;
  (*this)->attach_ivar = scope;
  (*this)->attach_stage = parent;
  bool found = false;
  for (size_t i = 0; i < parent->leaf_iter_vars.size(); ++i) {
    if (scope == parent->leaf_iter_vars[i]) {
      found = true;
      break;
    }
  }
  CHECK(found) << "Cannot find the axis " << scope << " in parent's leaf_iter_vars"
               << " parent=" << parent;
  return *this;
}

Stage& Stage::compute_inline() {  // NOLINT(*)
  CHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kInline;
  return *this;
}

Stage& Stage::compute_root() {  // NOLINT(*)
  CHECK_NE((*this)->attach_type, kScanUpdate) << "Cannot specify compute_at for scan updates";
  (*this)->attach_type = kGroupRoot;
  return *this;
}

Stage& Stage::bind(IterVar ivar, IterVar thread_ivar) {  // NOLINT(*)
  StageNode* self = operator->();
  CHECK(ivar->iter_type == kDataPar || ivar->iter_type == kCommReduce)
      << "Cannot bind " << IterVarType2String(ivar->iter_type) << " to thread";
  CHECK(thread_ivar->iter_type == kThreadIndex)
      << "Cannot rebase by " << IterVarType2String(ivar->iter_type)
      << ", only thread axis is allowed so far";
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, ivar);

  auto it = self->iter_var_attrs.find(ivar);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
    if (n->bind_thread.defined() && !n->bind_thread.same_as(thread_ivar)) {
      LOG(WARNING) << "Axis " << ivar << " is already bind to another thread " << n->bind_thread;
    }
  } else {
    n = make_object<IterVarAttrNode>();
  }
  n->bind_thread = thread_ivar;
  self->iter_var_attrs.Set(ivar, IterVarAttr(n));
  return *this;
}

Stage& Stage::env_threads(Array<IterVar> threads) {
  StageNode* self = operator->();
  CHECK(self->op.defined() && self->op.as<ScanOpNode>())
      << "env_threads is only valid for composite ops such as ScanOp";
  CHECK_EQ(self->env_threads.size(), 0U) << "Already set env_threads";
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;
  Array<IterVar>& all_vars = self->all_iter_vars;
  std::vector<ObjectRef> temp;
  for (IterVar iv : threads) {
    temp.push_back(iv);
  }
  leaf_vars.insert(leaf_vars.begin(), temp.begin(), temp.end());
  all_vars.insert(all_vars.end(), temp.begin(), temp.end());
  self->env_threads = threads;
  return *this;
}

Stage& Stage::set_store_predicate(PrimExpr predicate) {
  StageNode* self = operator->();
  self->store_predicate = predicate;
  return *this;
}

Stage& Stage::split(IterVar parent, PrimExpr factor, IterVar* p_outer,
                    IterVar* p_inner) {  // NOLINT(*)
  SplitHelper(operator->(), parent, factor, PrimExpr(), p_outer, p_inner);
  return *this;
}

Stage& Stage::split_by_nparts(IterVar parent, PrimExpr nparts, IterVar* p_outer,
                              IterVar* p_inner) {  // NOLINT(*)
  SplitHelper(operator->(), parent, PrimExpr(), nparts, p_outer, p_inner);
  return *this;
}

Stage& Stage::fuse(IterVar outer, IterVar inner, IterVar* p_target) {  // NOLINT(*)
  StageNode* self = operator->();
  CHECK(outer->iter_type == kDataPar || outer->iter_type == kCommReduce ||
        outer->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(outer->iter_type);
  CHECK(inner->iter_type == kDataPar || inner->iter_type == kCommReduce ||
        inner->iter_type == kOrdered)
      << "Cannot fuse " << IterVarType2String(inner->iter_type);

  IterVarType iter_type = outer->iter_type;
  if (inner->iter_type > iter_type) iter_type = inner->iter_type;
  std::string fused_name = outer->var->name_hint + "." + inner->var->name_hint + ".fused";
  DataType iter_dtype = MatchDataType({inner->var.dtype(), outer->var.dtype()});

  IterVar fused = IterVar(Range(), Var(fused_name, iter_dtype), iter_type);

  Array<IterVar>& all_vars = self->all_iter_vars;
  Array<IterVar>& leaf_vars = self->leaf_iter_vars;

  size_t pos_inner = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), inner);
  size_t pos_outer = FindLeafVar(all_vars.GetArrayNode(), leaf_vars.GetArrayNode(), outer);
  if (pos_inner + 1 == pos_outer) {
    std::swap(outer, inner);
    std::swap(pos_inner, pos_outer);
  }
  CHECK_EQ(pos_inner, pos_outer + 1)
      << "Can only fuse iterations that are consecutive between each other";
  self->relations.push_back(Fuse(outer, inner, fused));
  all_vars.push_back(fused);
  leaf_vars.erase(leaf_vars.begin() + pos_outer, leaf_vars.begin() + pos_inner + 1);
  leaf_vars.insert(leaf_vars.begin() + pos_outer, fused);
  *p_target = fused;
  return *this;
}

Stage& Stage::fuse(const Array<IterVar>& axes, IterVar* p_target) {  // NOLINT(*)
  if (axes.size() != 0) {
    IterVar fused = axes[0];
    for (size_t i = 1; i < axes.size(); ++i) {
      this->fuse(fused, axes[i], &fused);
    }
    *p_target = std::move(fused);
  } else {
    StageNode* self = operator->();
    // special handle fuse empty array.
    // insert at the outer most loop
    IterVar singleton =
        IterVar(Range::FromMinExtent(0, 1), Var("singleton", DataType::Int(32)), kDataPar);
    self->relations.push_back(Singleton(singleton));
    Array<IterVar>& all_vars = self->all_iter_vars;
    Array<IterVar>& leaf_vars = self->leaf_iter_vars;
    all_vars.push_back(singleton);
    leaf_vars.insert(leaf_vars.begin(), singleton);
    *p_target = singleton;
  }
  return *this;
}

Stage& Stage::reorder(const Array<IterVar>& order) {  // NOLINT(*)
  std::unordered_set<IterVar> seen_var;
  StageNode* self = operator->();
  for (IterVar iv : order) {
    CHECK(iv->iter_type == kDataPar || iv->iter_type == kCommReduce ||
          iv->iter_type == kThreadIndex)
        << "Cannot reorder IterVar(" << IterVarType2String(iv->iter_type) << ")";

    CHECK_EQ(seen_var.count(iv), 0) << "Same axis can not appear more than once " << iv;
    seen_var.insert(iv);
  }
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  std::vector<size_t> pos;

  for (size_t i = 0; i < order.size(); ++i) {
    pos.push_back(FindLeafVar(all_vars, leaf_vars, order[i]));
  }
  std::vector<ObjectRef> temp;
  for (size_t i = 0; i < pos.size(); ++i) {
    temp.emplace_back(leaf_vars->at(pos[i]));
  }
  std::sort(pos.begin(), pos.end());
  for (size_t i = 0; i < pos.size(); ++i) {
    leaf_vars->SetItem(pos[i], temp[i]);
  }
  return *this;
}

Stage& Stage::tile(IterVar x_parent, IterVar y_parent, PrimExpr x_factor, PrimExpr y_factor,
                   IterVar* p_x_outer, IterVar* p_y_outer, IterVar* p_x_inner, IterVar* p_y_inner) {
  split(x_parent, x_factor, p_x_outer, p_x_inner);
  split(y_parent, y_factor, p_y_outer, p_y_inner);
  reorder(Array<IterVar>({*p_x_outer, *p_y_outer, *p_x_inner, *p_y_inner}));
  return *this;
}

template <typename FUpdate>
inline void UpdateIterVarAttr(StageNode* self, IterVar var, FUpdate fupdate,
                              bool need_leaf = true) {
  if (need_leaf) {
    ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
    ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
    FindLeafVar(all_vars, leaf_vars, var);
  }
  auto it = self->iter_var_attrs.find(var);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_object<IterVarAttrNode>();
  }
  fupdate(n.get());
  self->iter_var_attrs.Set(var, IterVarAttr(n));
}

inline void SetAttrIterType(StageNode* self, IterVar var, IterVarType iter_type) {
  UpdateIterVarAttr(self, var, [iter_type](IterVarAttrNode* n) { n->iter_type = iter_type; });
}

Stage& Stage::vectorize(IterVar var) {  // NOLINT(*)
  CHECK(var->iter_type == kDataPar || var->iter_type == kOpaque || var->iter_type == kUnrolled ||
        var->iter_type == kVectorized || var->iter_type == kTensorized ||
        var->iter_type == kParallelized)
      << "Cannot vectorize on " << IterVarType2String(var->iter_type);
  SetAttrIterType(operator->(), var, kVectorized);
  return *this;
}

Stage& Stage::tensorize(IterVar var, TensorIntrin f) {  // NOLINT(*)
  UpdateIterVarAttr(operator->(), var, [f](IterVarAttrNode* n) {
    n->iter_type = kTensorized;
    n->tensor_intrin = f;
  });
  return *this;
}

Stage& Stage::unroll(IterVar var) {  // NOLINT(*)
  SetAttrIterType(operator->(), var, kUnrolled);
  return *this;
}

Stage& Stage::parallel(IterVar var) {  // NOLINT(*)
  SetAttrIterType(operator->(), var, kParallelized);
  return *this;
}

Stage& Stage::pragma(IterVar var, const std::string& pragma_type,
                     const PrimExpr& pragma_value) {  // NOLINT(*)
  if (pragma_type == "unroll") {
    this->unroll(var);
  } else if (pragma_type == "vectorize") {
    this->vectorize(var);
  } else {
    UpdateIterVarAttr(operator->(), var, [pragma_type, pragma_value](IterVarAttrNode* n) {
      n->pragma_keys.push_back(tir::StringImm(pragma_type));
      n->pragma_values.push_back(pragma_value);
    });
  }
  return *this;
}

Stage& Stage::prefetch(const Tensor& tensor, IterVar var, PrimExpr offset) {
  StageNode* self = operator->();
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  FindLeafVar(all_vars, leaf_vars, var);
  auto it = self->iter_var_attrs.find(var);
  ObjectPtr<IterVarAttrNode> n;
  if (it != self->iter_var_attrs.end()) {
    n = make_object<IterVarAttrNode>(*(*it).second.operator->());
  } else {
    n = make_object<IterVarAttrNode>();
  }
  n->prefetch_data.push_back(tensor);
  n->prefetch_offset.push_back(offset);
  self->iter_var_attrs.Set(var, IterVarAttr(n));
  return *this;
}

Stage& Stage::storage_align(IterVar axis, int factor, int offset) {
  StageNode* self = operator->();
  UpdateIterVarAttr(self, axis,
                    [factor, offset](IterVarAttrNode* n) {
                      n->dim_align_factor = factor;
                      n->dim_align_offset = offset;
                    },
                    false);
  return *this;
}

Stage& Stage::double_buffer() {
  StageNode* self = operator->();
  CHECK(!self->is_output) << "Cannot apply double buffer on output";
  self->double_buffer = true;
  return *this;
}

//////////////DEBUG//////////////////
void printreadgraph(te::ReadGraph rg) {
  std::cout << "READGRAPH:" << std::endl;
  for (auto kv : rg) {
    std::cout << kv.first << ":";
    for (auto v : kv.second) {
      std::cout << v << ",";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

/////////////////DEBUG/////////////

#if TSL_DBG_V1
Stage& Stage::decompose(Array<PrimExpr> factors, Array<IterVar>& ret_ivars) { arith::Analyzer ana; }
#endif

#if TSL_DBG_V0

void decompStackPushHelper(StageNode* self, const Array<PrimExpr>& factors,
                           const Array<IterVar>& new_axis) {
  auto& stacktop = self->decomp_stack.back();
  CHECK_EQ(factors.size(), stacktop.left_ivars.size());
  auto& leaf_vars = self->leaf_iter_vars;
  StageNode::DecompEntry entry;
  entry.factors = factors;
  entry.level = stacktop.level + 1;
  for (size_t i = 0; i < stacktop.left_ivars.size(); i++) {
    auto& v = stacktop.left_ivars[i];
    CHECK(v->iter_type != kCommReduce) << "reduce not supported yet";
    size_t pos = FindNodeRef(leaf_vars.GetArrayNode(), v);
    if (pos < leaf_vars.size()) {
      leaf_vars.erase(leaf_vars.begin() + pos);
    } else {
      LOG(FATAL);
    }
    std::ostringstream os;
    os << "." << self->decomp_stack.size();
    auto prefix = os.str();
    IterVar left = IterVar(Range(), v->var.copy_with_suffix(prefix + "L"), v->iter_type);
    IterVar right = IterVar(Range(), v->var.copy_with_suffix(prefix + "R"), v->iter_type);
    self->leaf_iter_vars.push_back(left);
    self->leaf_iter_vars.push_back(right);
    self->all_iter_vars.push_back(left);
    self->all_iter_vars.push_back(right);
    auto split = Split(v, right, left, PrimExpr(), indexdiv(stacktop.factors[i], factors[i]));
    self->relations.push_back(split);
    entry.left_ivars.push_back(left);
    entry.right_ivars.push_back(right);
    entry.split_relations.push_back(split);
  }
  self->decomp_stack.push_back(entry);
  auto& stackbottom = self->decomp_stack.front();
  CHECK(stackbottom.split_relations.size() ==
        new_axis.size());  // TODO: this is true if no reduce exists in original op
  for (size_t i = 0; i < new_axis.size(); i++) {
    auto& old_relation = stackbottom.split_relations[i];
    auto new_relation = Split(new_axis[i], old_relation->outer, old_relation->inner,
                              old_relation->factor, old_relation->nparts);
    size_t pos = FindNodeRef(self->relations.GetArrayNode(), old_relation);
    if (pos < self->relations.size()) {
      self->relations.erase(self->relations.begin() + pos);
      self->relations.insert(self->relations.begin() + pos, new_relation);
    } else {
      LOG(FATAL);
    }
    stackbottom.split_relations.erase(stackbottom.split_relations.begin() + i);
    stackbottom.split_relations.insert(stackbottom.split_relations.begin() + i, new_relation);
  }
}

Stage& Stage::decompose(Array<PrimExpr> factors, Array<IterVar>& ret_ivars) {
  StageNode* self = operator->();
  std::cout << "DECOMPOSING:" << self->op << std::endl;
  Operation self_op = self->op;
  arith::Analyzer ana;
  if (auto* origin_op = self_op.as<ComputeOpNode>()) {
    CHECK(origin_op->attrs.count("TslOp") != 0) << "Not a TslOp";

    auto& stacktop = self->decomp_stack.back();
    Array<PrimExpr> origin_shape = origin_op->output_shape(0);
    Array<IterVar> origin_axis = origin_op->axis;
    CHECK(factors.size() == origin_shape.size()) << "shape not coherent";

    // generate new in u/eshape
    Array<PrimExpr> new_out_eshape, new_out_ushape;
    for (size_t i = 0; i < factors.size(); i++) {
      CHECK(ana.CanProve(factors[i] < stacktop.factors[i]))
          << "greater decomposition factor in nested decompose not allowed";
      CHECK(ana.CanProve(indexmod(stacktop.factors[i], factors[i]) == 0))
          << "Non-even decomposition not allowed";
      new_out_eshape.push_back(ana.Simplify(factors[i]));
      new_out_ushape.push_back(ana.Simplify(indexdiv(origin_shape[i], factors[i])));
    }
    // stage 1: generate new traget decomposing op
    Array<IterVar> new_axis;
    for (size_t i = 0; i < origin_axis.size(); i++) {
      new_axis.push_back(IterVar(Range(0, new_out_ushape[i]), origin_axis[i]->var,
                                 origin_axis[i]->iter_type, origin_axis[i]->thread_tag));
    }
    auto new_body = Downcast<Array<TslExpr>>(origin_op->body);
    auto new_op = ComputeOp(origin_op->name, origin_op->tag, origin_op->attrs, new_axis,
                            origin_shape, new_out_ushape, new_out_eshape, origin_op->in_ushape,
                            origin_op->out_eshape, new_body);

    // stage 2: handle stage's itervar information properly
    //!!!!CURRENTLY, SHOULD ALWAYS DECOMPOSE BEFORE MAKING ANY OTHER SCHEDULING!!!
    decompStackPushHelper(self, factors, new_axis);
    ret_ivars = self->decomp_stack.back().right_ivars;

    // stage 3: update stage's op and update reader operations
    self->op = new_op;
    std::cout << "NEW:" << new_op.output(0) << std::endl;

    auto readgraph =
        CreateReadGraph(self->parent_sched->outputs);  // no new op is added, so readgraph of
                                                       // original schedule is enough
    auto feedgraph = CreateFeedGraph(readgraph);
    te::Tensor origin_tensor = self->origin_op.output(
        0);  // only meant to use original op's tensor output to find readers!
    te::Tensor new_tensor = new_op.output(0);
    if (feedgraph.find(origin_tensor) != feedgraph.end()) {
      auto readers = feedgraph.at(origin_tensor);
      std::unordered_map<Tensor, Tensor> vsub;
      vsub[self_op.output(0)] = new_tensor;
      std::unordered_map<Tensor, Tensor> vmap;
      std::unordered_map<Tensor, Tensor> rvmap;
      for (Operation op : readers) {
        Stage s = operator->()->parent_sched->stage_map[op];
        Operation repl_op = s->op->ReplaceInputs(s->op, vsub);
        CHECK(!repl_op.same_as(s->op))
            << "Cannot find " << origin_tensor << " in the inputs of " << s->op;
        vmap[s->op.output(0)] = repl_op.output(0);
        rvmap[repl_op.output(0)] = s->op.output(0);
        s->op = repl_op;
      }
      ReplaceDataFlow(self->parent_sched->stages, &vmap,
                      &rvmap);  // use this TVM infra to make all possible updates in dataflow
                                // graph(stage graph)
    }

    // stage 3e: some debug info.
    readgraph = CreateReliableReadGraph(self->parent_sched);
    std::cout << "////////////////////////////" << std::endl;
    printreadgraph(readgraph);

    Array<Operation> updated_outputs;
    for (auto& v : self->parent_sched->outputs) {
      updated_outputs.push_back(self->parent_sched->stage_map[v]->op);
    }
    auto postdfs = PostDFSOrder(updated_outputs, readgraph);
    for (auto& v : postdfs) {
      std::cout << v.output(0) << std::endl;
    }

    // stage 4: backward elemshape inference
    // CURRENT VERSION HAS NO SUPPORT FOR REDUCE. TODO: implement proposed infinite/finite reduce
    // and investigate x+rx

  } else {
    CHECK(0) << "Can only decompose ComputeOp";
  }

  return *this;
}  // namespace te

#endif

#if TSL_DBG_V1
Stage& Stage::decompose(Array<PrimExpr> factors, Array<IterVar>& ret_ivars) {

}
#endif

    Stage CopyStage(const Stage& s) {
  ObjectPtr<StageNode> n = make_object<StageNode>(*s.operator->());
  return Stage(n);
}

Schedule Schedule::copy() const {
  // map of stages.
  const ScheduleNode* self = operator->();
  std::unordered_map<Stage, Stage, ObjectPtrHash, ObjectPtrEqual> smap;
  ObjectPtr<ScheduleNode> n = make_object<ScheduleNode>();
  n->outputs = self->outputs;
  // Copy the stages.
  for (Stage s : self->stages) {
    Stage scopy = CopyStage(s);
    smap[s] = scopy;
    n->stages.push_back(scopy);
  }
  for (Stage g : self->groups) {
    Stage gcopy = CopyStage(g);
    smap[g] = gcopy;
    n->groups.push_back(gcopy);
  }
  // Remaps the reference relations.
  for (auto kv : self->stage_map) {
    n->stage_map.Set(kv.first, smap.at(kv.second));
  }
  for (Stage s : n->stages) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  for (Stage s : n->groups) {
    if (s->attach_stage.defined()) {
      CHECK(smap.find(s->attach_stage) != smap.end())
          << s->attach_stage << " not found in " << (*this);
      s->attach_stage = smap.at(s->attach_stage);
    }
    if (s->group.defined()) {
      CHECK(smap.find(s->group) != smap.end()) << s->group << " not found in " << (*this);
      s->group = smap.at(s->group);
    }
  }
  return Schedule(n);
}

Stage Schedule::operator[](const Operation& op) {
  auto it = (*this)->stage_map.find(op);
  CHECK(it != (*this)->stage_map.end())
      << "Cannot find Stage for operator " << op << " in the schedule";
  return (*it).second;
}

Stage LeastCommonAncestor(Stage g1, Stage g2) {
  if (!g1.defined()) return g1;
  if (!g2.defined()) return g2;
  if (g1.same_as(g2)) return g1;
  Stage g = g1;
  while (g.defined()) {
    if (g.same_as(g2)) return g2;
    g = g->group;
  }
  g = g2;
  while (g.defined()) {
    if (g.same_as(g1)) return g1;
    g = g->group;
  }
  return g;
}

Array<Tensor> RemapTensor(ScheduleNode* self, const Array<Tensor>& arr) {
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  Array<Tensor> ret;
  for (Tensor t : arr) {
    if (!op2stage_cache.count(t->op.get())) {
      CHECK(self->stage_map.count(t->op)) << "Given tensor is not in the schedule plan";
      t = self->stage_map[t->op]->op.output(t->value_index);
    }
    ret.push_back(t);
  }
  return ret;
}

// Group the schedule stages.
Stage Schedule::create_group(const Array<Tensor>& outputs, const Array<Tensor>& inputs,
                             bool include_inputs) {
  ScheduleNode* self = operator->();
  self->InitCache();
  const auto& op2stage_cache = self->op2stage_cache_;
  // Get the ops.
  Array<Operation> ops =
      te::GetSubGraph(RemapTensor(self, outputs), RemapTensor(self, inputs), include_inputs);
  // local counter entry
  // Automatically initialize to 0 during creation.
  struct Entry {
    int count{0};
  };
  // Map of group->touched counter
  std::unordered_map<Stage, Entry, ObjectPtrHash, ObjectPtrEqual> counter;
  // The parent group;
  Stage parent_group;
  // Detect common parent and child.
  for (size_t i = 0; i < ops.size(); ++i) {
    Operation op = ops[i];
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage op_group = it->second->group;
    if (i == 0) {
      parent_group = op_group;
    } else {
      parent_group = LeastCommonAncestor(parent_group, op_group);
    }
    if (op_group.defined()) {
      ++counter[op_group].count;
    }
  }
  // Create the new group stage.
  Stage gstage(make_object<StageNode>());
  gstage->group = parent_group;
  if (parent_group.defined()) {
    ++parent_group->num_child_stages;
  }
  // Propagate the counter statistics from by checking if subgroup
  // Is full and propagate.
  std::vector<Stage> stack;
  for (auto& kv : counter) {
    if (!kv.first.same_as(parent_group)) {
      if (kv.first->num_child_stages == kv.second.count) {
        stack.push_back(kv.first);
      }
    }
  }
  while (!stack.empty()) {
    Stage g = stack.back();
    stack.pop_back();
    if (g->group.defined() && !g->group.same_as(parent_group)) {
      Entry& e = counter[g->group];
      ++e.count;
      if (e.count == g->group->num_child_stages) {
        stack.push_back(g->group);
      }
    }
  }
  // Verification and remappig the subgroups.
  for (auto& kv : counter) {
    if (kv.first.same_as(parent_group)) continue;
    CHECK_EQ(kv.first->num_child_stages, kv.second.count)
        << "Trying to group region that intersect with an already existed group";
    if (kv.first->group.same_as(parent_group)) {
      Stage s = kv.first;
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Remap the group of op stages.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->group.same_as(parent_group)) {
      s->group = gstage;
      ++gstage->num_child_stages;
      if (parent_group.defined()) {
        --parent_group->num_child_stages;
      }
    }
  }
  // Correct the attach to keep everything in group.
  for (Operation op : ops) {
    auto it = op2stage_cache.find(op.get());
    CHECK(it != op2stage_cache.end());
    Stage s = it->second;
    if (s->attach_type == kScope) {
      Stage cg = LeastCommonAncestor(s->attach_stage->group, gstage);
      if (!cg.same_as(gstage)) {
        LOG(WARNING) << "group invalidates some previous compute_at relation "
                     << " and keeps things to be computed inside the group";
        s.compute_root();
      }
    }
  }

  self->groups.push_back(gstage);
  return gstage;
}

void ScheduleNode::InvalidateCache() { op2stage_cache_.clear(); }

void ScheduleNode::InitCache() {
  if (op2stage_cache_.size() == stages.size()) return;
  InvalidateCache();
  for (Stage s : stages) {
    if (s->op.defined()) {
      op2stage_cache_[s->op.get()] = s;
    }
  }
  CHECK_EQ(op2stage_cache_.size(), stages.size());
}

bool ScheduleNode::Contain(const Operation& op) const {
  return stage_map.find(op) != stage_map.end();
}

Schedule::Schedule(Array<Operation> ops) {
  auto n = make_object<ScheduleNode>();
  data_ = n;
  n->outputs = ops;
  auto g = te::CreateReadGraph(n->outputs);
  Array<Operation> post_order = te::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage;
    #if TSL_DBG_V0
    if (op->attrs.count("TslOp") != 0) {
      stage = Stage(op, this->operator->());  // xjx: tsl modification, let stagenode hold a
                                              // scheduleNode* pointer, by this we can call
                                              // createreliablereadGraph inside stage methods.
      stage->is_tsl_stage = true;
    } else {
      stage = Stage(op);
    }
    #else 
    stage = Stage(op);
    #endif
    stage->is_output = output_set.count(op) != 0;
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
    // mark scan updates.
    if (const ScanOpNode* scan = op.as<ScanOpNode>()) {
      Array<Tensor> inputs;
      for (Tensor t : scan->state_placeholder) {
        inputs.push_back(t);
      }
      for (Tensor t : scan->inputs) {
        inputs.push_back(t);
      }
      // Create the scan group.
      Stage scan_group = this->create_group(scan->update, inputs, false);
      scan_group->attach_type = kScanUpdate;
      scan_group->attach_stage = stage;

      for (size_t i = 0; i < scan->update.size(); ++i) {
        Stage s = n->stage_map[scan->update[i]->op];
        CHECK(scan_group.same_as(s->group));
      }
    }
  }
}

Split::Split(IterVar parent, IterVar outer, IterVar inner, PrimExpr factor, PrimExpr nparts) {
  auto n = make_object<SplitNode>();
  n->parent = parent;
  n->outer = outer;
  n->inner = inner;
  n->factor = factor;
  n->nparts = nparts;
  data_ = std::move(n);
}

Fuse::Fuse(IterVar outer, IterVar inner, IterVar fused) {
  auto n = make_object<FuseNode>();
  n->outer = outer;
  n->inner = inner;
  n->fused = fused;
  data_ = std::move(n);
}

Rebase::Rebase(IterVar parent, IterVar rebased) {
  auto n = make_object<RebaseNode>();
  n->parent = parent;
  n->rebased = rebased;
  data_ = std::move(n);
}

Singleton::Singleton(IterVar iter) {
  auto n = make_object<SingletonNode>();
  n->iter = iter;
  data_ = std::move(n);
}

SpecializedCondition::SpecializedCondition(Array<PrimExpr> conditions) {
  ObjectPtr<SpecializedConditionNode> n = make_object<SpecializedConditionNode>();
  n->clauses = std::move(conditions);
  data_ = std::move(n);
}

/*! \brief Entry to hold the SpecializedCondition context stack. */
struct TVMSpecializationThreadLocalEntry {
  /*! \brief The current specialized condition */
  std::stack<SpecializedCondition> condition_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
typedef dmlc::ThreadLocalStore<TVMSpecializationThreadLocalEntry> TVMSpecializationThreadLocalStore;

void SpecializedCondition::EnterWithScope() {
  TVMSpecializationThreadLocalEntry* entry = TVMSpecializationThreadLocalStore::Get();
  entry->condition_stack.push(*this);
}

void SpecializedCondition::ExitWithScope() {
  TVMSpecializationThreadLocalEntry* entry = TVMSpecializationThreadLocalStore::Get();
  CHECK(!entry->condition_stack.empty());
  CHECK(entry->condition_stack.top().same_as(*this));
  entry->condition_stack.pop();
}

SpecializedCondition SpecializedCondition::Current() {
  TVMSpecializationThreadLocalEntry* entry = TVMSpecializationThreadLocalStore::Get();
  SpecializedCondition cond;
  if (entry->condition_stack.size() > 0) {
    cond = entry->condition_stack.top();
  }
  return cond;
}

class SpecializedCondition::Internal {
 public:
  static void EnterScope(SpecializedCondition cond) { cond.EnterWithScope(); }

  static void ExitScope(SpecializedCondition cond) { cond.ExitWithScope(); }
};

TVM_REGISTER_NODE_TYPE(StageNode);
TVM_REGISTER_NODE_TYPE(IterVarAttrNode);
TVM_REGISTER_NODE_TYPE(SplitNode);
TVM_REGISTER_NODE_TYPE(FuseNode);
TVM_REGISTER_NODE_TYPE(RebaseNode);
TVM_REGISTER_NODE_TYPE(SingletonNode);
TVM_REGISTER_NODE_TYPE(ScheduleNode);
TVM_REGISTER_NODE_TYPE(SpecializedConditionNode);

// Printer
TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StageNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StageNode*>(node.get());
      if (op->op.defined()) {
        p->stream << "stage(" << op->origin_op->name << ", " << op << ")";
      } else {
        p->stream << "group-stage(" << op << ")";
      }
    })
    .set_dispatch<IterVarAttrNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IterVarAttrNode*>(node.get());
      p->stream << IterVarType2String(op->iter_type);
    })
    .set_dispatch<SplitNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SplitNode*>(node.get());
      p->stream << "split(parent=";
      p->Print(op->parent);
      p->stream << ", outer=";
      p->Print(op->outer);
      p->stream << ", inner=";
      p->Print(op->inner);
      p->stream << ')';
    })
    .set_dispatch<FuseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const FuseNode*>(node.get());
      p->stream << "split(";
      p->stream << "outer=";
      p->Print(op->outer);
      p->stream << ", inner=";
      p->Print(op->inner);
      p->stream << ", fused=";
      p->Print(op->fused);
      p->stream << ')';
    })
    .set_dispatch<RebaseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const RebaseNode*>(node.get());
      p->stream << "rebase(";
      p->stream << "parent=";
      p->Print(op->parent);
      p->stream << ", rebased=";
      p->Print(op->rebased);
      p->stream << ')';
    })
    .set_dispatch<SingletonNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SingletonNode*>(node.get());
      p->stream << "singleton(";
      p->Print(op->iter);
      p->stream << ')';
    })
    .set_dispatch<ScheduleNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ScheduleNode*>(node.get());
      p->stream << "schedule(" << op << ")";
    })
    .set_dispatch<SpecializedConditionNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SpecializedConditionNode*>(node.get());
      p->stream << "specialized_condition(";
      p->Print(op->clauses);
      p->stream << ')';
    });

TVM_REGISTER_GLOBAL("te.CreateSchedule").set_body_typed(create_schedule);

TVM_REGISTER_GLOBAL("te.StageSetScope").set_body_method(&Stage::set_scope);

TVM_REGISTER_GLOBAL("te.StageBind").set_body_method(&Stage::bind);

TVM_REGISTER_GLOBAL("te.StageSplitByFactor")
    .set_body_typed([](Stage stage, IterVar parent, PrimExpr factor) {
      IterVar outer, inner;
      stage.split(parent, factor, &outer, &inner);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("te.StageSplitByNParts")
    .set_body_typed([](Stage stage, IterVar parent, PrimExpr nparts) {
      IterVar outer, inner;
      stage.split_by_nparts(parent, nparts, &outer, &inner);
      return Array<IterVar>({outer, inner});
    });

TVM_REGISTER_GLOBAL("te.StageFuse").set_body_typed([](Stage stage, Array<IterVar> axes) {
  IterVar fused;
  stage.fuse(axes, &fused);
  return fused;
});

TVM_REGISTER_GLOBAL("te.StageComputeAt").set_body_method(&Stage::compute_at);

TVM_REGISTER_GLOBAL("te.StageComputeInline").set_body_method(&Stage::compute_inline);

TVM_REGISTER_GLOBAL("te.StageComputeRoot").set_body_method(&Stage::compute_root);

TVM_REGISTER_GLOBAL("te.StageReorder").set_body_method(&Stage::reorder);

TVM_REGISTER_GLOBAL("te.StageTile")
    .set_body_typed([](Stage stage, IterVar x_parent, IterVar y_parent, PrimExpr x_factor,
                       PrimExpr y_factor) {
      IterVar x_outer, y_outer, x_inner, y_inner;
      stage.tile(x_parent, y_parent, x_factor, y_factor, &x_outer, &y_outer, &x_inner, &y_inner);
      return Array<IterVar>({x_outer, y_outer, x_inner, y_inner});
    });

TVM_REGISTER_GLOBAL("te.StageEnvThreads").set_body_method(&Stage::env_threads);

TVM_REGISTER_GLOBAL("te.StageSetStorePredicate").set_body_method(&Stage::set_store_predicate);

TVM_REGISTER_GLOBAL("te.StageUnroll").set_body_method(&Stage::unroll);

TVM_REGISTER_GLOBAL("te.StageVectorize").set_body_method(&Stage::vectorize);

TVM_REGISTER_GLOBAL("te.StageTensorize").set_body_method(&Stage::tensorize);

TVM_REGISTER_GLOBAL("te.StageParallel").set_body_method(&Stage::parallel);

TVM_REGISTER_GLOBAL("te.StagePragma").set_body_method(&Stage::pragma);

TVM_REGISTER_GLOBAL("te.StagePrefetch").set_body_method(&Stage::prefetch);

TVM_REGISTER_GLOBAL("te.StageStorageAlign").set_body_method(&Stage::storage_align);

TVM_REGISTER_GLOBAL("te.StageDoubleBuffer").set_body_method(&Stage::double_buffer);

TVM_REGISTER_GLOBAL("te.ScheduleNormalize").set_body_method(&Schedule::normalize);

TVM_REGISTER_GLOBAL("te.ScheduleCreateGroup").set_body_method(&Schedule::create_group);

TVM_REGISTER_GLOBAL("te.ScheduleCacheRead").set_body_method(&Schedule::cache_read);

TVM_REGISTER_GLOBAL("te.ScheduleCacheWrite").set_body([](TVMArgs args, TVMRetValue* ret) {
  if (args[1].IsObjectRef<Tensor>()) {
    *ret = args[0].operator Schedule().cache_write(args[1].operator Tensor(), args[2]);
  } else {
    *ret = args[0].operator Schedule().cache_write(args[1].operator Array<Tensor>(), args[2]);
  }
});

TVM_REGISTER_GLOBAL("te.ScheduleRFactor").set_body_method(&Schedule::rfactor);

TVM_REGISTER_GLOBAL("te.CreateSpecializedCondition").set_body_typed([](Array<PrimExpr> condition) {
  return SpecializedCondition(condition);
});

TVM_REGISTER_GLOBAL("te.GetCurrentSpecialization").set_body([](TVMArgs args, TVMRetValue* ret) {
  *ret = SpecializedCondition::Current();
});

TVM_REGISTER_GLOBAL("te.EnterSpecializationScope")
    .set_body_typed(SpecializedCondition::Internal::EnterScope);

TVM_REGISTER_GLOBAL("te.ExitSpecializationScope")
    .set_body_typed(SpecializedCondition::Internal::ExitScope);

}  // namespace te
}  // namespace tvm
