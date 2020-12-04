// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/fuse_conv_bn_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

#define GET_CONV_BN_NODES(pattern_name)                                      \
  /* OPERATORS */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(conv, conv, pattern_name);                       \
  GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, pattern_name);           \
  /* CONV inputs */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, pattern_name);         \
  /* CONV outputs */                                                         \
  GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, pattern_name);               \
  /* BN inputs */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, pattern_name);               \
  GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, pattern_name);         \
  /* BN outputs */                                                           \
  GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, pattern_name); /* Out */         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, pattern_name);         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out, pattern_name); \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, pattern_name);     \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance, pattern_name)

void copy_tensor_to_gpu(framework::LoDTensor* in_tensor) {
  platform::CPUPlace cpu_place;
  platform::CUDAPlace gpu_place(0);
  framework::LoDTensor temp_tensor;
  temp_tensor.Resize(in_tensor->dims());
  temp_tensor.mutable_data<float>(cpu_place);

  // Copy the parameter data to a tmp tensor.
  TensorCopySync(*in_tensor, cpu_place, &temp_tensor);
  // Reallocation the space on GPU
  in_tensor->clear();
  // Copy parameter data to newly allocated GPU space.
  TensorCopySync(temp_tensor, gpu_place, in_tensor);
}

void copy_tensor_to_cpu(framework::LoDTensor* in_tensor) {
  platform::CPUPlace cpu_place;
  framework::LoDTensor temp_tensor;
  temp_tensor.Resize(in_tensor->dims());
  temp_tensor.mutable_data<float>(cpu_place);

  // Copy the parameter data to a tmp tensor.
  TensorCopySync(*in_tensor, cpu_place, &temp_tensor);
  // Reallocation the space on cpu
  in_tensor->clear();
  // Copy parameter data to newly allocated cpu space.
  TensorCopySync(temp_tensor, cpu_place, in_tensor);
}

static void VisualizeGraph(Graph* graph, std::string graph_viz_path) {
  // Insert a graph_viz_pass to transform the graph to a .dot file.
  // It can be used for debug.
  auto graph_viz_pass = PassRegistry::Instance().Get("graph_viz_pass");
  graph_viz_pass->Set("graph_viz_path", new std::string(graph_viz_path));
  graph_viz_pass->Apply(graph);
}

static void recompute_bias_and_weights(const Scope* scope,
                                       ir::Node* conv_weight,            //
                                       const ir::Node& bn_scale,         //
                                       const LoDTensor& bn_bias_tensor,  //
                                       const ir::Node& bn_mean,          //
                                       const ir::Node& bn_variance,      //
                                       LoDTensor* eltwise_y_in_tensor,   //
                                       float epsilon,
                                       const std::string& conv_type) {
  using EigenVectorArrayMap =
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
  using ConstEigenVectorArrayMap =
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
  using EigenMatrixArrayMap = Eigen::Map<
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Re-compute bias of conv2d from BN
  PADDLE_ENFORCE_EQ(
      eltwise_y_in_tensor->dims(), bn_bias_tensor.dims(),
      platform::errors::InvalidArgument("Tensor elementwise y(%d) and batch "
                                        "norm bias(%d) must have same dims.",
                                        eltwise_y_in_tensor->dims().size(),
                                        bn_bias_tensor.dims().size()));

  auto* scale_tensor = scope->FindVar(bn_scale.Name())->GetMutable<LoDTensor>();
  auto* variance_tensor =
      scope->FindVar(bn_variance.Name())->GetMutable<LoDTensor>();
  auto* mean_tensor = scope->FindVar(bn_mean.Name())->GetMutable<LoDTensor>();
  LOG(INFO) << "copy scale_tensor";
  copy_tensor_to_cpu(scale_tensor);
  LOG(INFO) << "copy variance_tensor";
  copy_tensor_to_cpu(variance_tensor);
  LOG(INFO) << "copy mean_tensor";
  copy_tensor_to_cpu(mean_tensor);
  LOG(INFO) << "copy eltwise_y_in_tensor";
  copy_tensor_to_cpu(eltwise_y_in_tensor);

  ConstEigenVectorArrayMap scale_array(scale_tensor->data<float>(),
                                       scale_tensor->numel(), 1);
  EigenVectorArrayMap variance_array(
      variance_tensor->mutable_data<float>(platform::CPUPlace()),
      variance_tensor->numel(), 1);
  ConstEigenVectorArrayMap mean_array(mean_tensor->data<float>(),
                                      mean_tensor->numel(), 1);
  ConstEigenVectorArrayMap bn_bias_array(bn_bias_tensor.data<float>(),
                                         bn_bias_tensor.numel(), 1);

  // variance will not be used anymore, so make it std_array and then tmp_array
  variance_array += epsilon;
  variance_array = variance_array.sqrt();
  variance_array = scale_array / variance_array;

  EigenVectorArrayMap eltwise_y_in_array(
      eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
      eltwise_y_in_tensor->numel(), 1);

  eltwise_y_in_array =
      ((eltwise_y_in_array - mean_array) * variance_array) + bn_bias_array;

  // Re-compute weight of conv2d from BN
  if (scope->FindVar(conv_weight->Name()) == nullptr) {
    LOG(INFO) << "Not find " << conv_weight->Name();
  }
  auto* weights = scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
  copy_tensor_to_cpu(weights);
  auto weights_shape = weights->dims();
  auto weights_data = weights->mutable_data<float>(platform::CPUPlace());

  // ConvTranspose weights are in IOHW format
  if (conv_type == "conv2d_transpose") {
    int kernel_size = weights_shape[2] * weights_shape[3];
    for (int i = 0; i < weights->numel();) {
      for (int j = 0; j < weights_shape[1]; ++j) {
        for (int k = 0; k < kernel_size; ++k, ++i) {
          weights_data[i] *= variance_array[j];
        }
      }
    }
  } else {
    auto weights_shape_2d = flatten_to_2d(weights_shape, 1);

    EigenMatrixArrayMap weights_array_2d(weights_data, weights_shape_2d[0],
                                         weights_shape_2d[1]);

    weights_array_2d.colwise() *= variance_array;
  }
  copy_tensor_to_gpu(weights);
  copy_tensor_to_gpu(eltwise_y_in_tensor);
}

void FuseConvBNPass::ApplyImpl(ir::Graph* graph) const {
  LOG(INFO) << "FuseConvBN";
  VisualizeGraph(graph, "./1.dot");
  graph = FuseConvBN(graph);
  VisualizeGraph(graph, "./2.dot");
  for (auto opt_type : {"sgd", "momentum"}) {
    for (auto flag : {true, false}) {
      LOG(INFO) << "FuseConvBNGrad " << opt_type << " " << flag;
      graph = FuseConvBNGrad(graph, opt_type, flag);
    }
  }
  VisualizeGraph(graph, "./3.dot");
}

ir::Graph* FuseConvBNPass::FuseConvBN(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  // auto* scope = Get<const std::vector<Scope *>>(details::kLocalScopes)[0];
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input(conv_type(), "Input");
  patterns::ConvBN conv_bn_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bn_pattern(conv_input, conv_type(), false /*with_eltwise_add*/);

  int found_conv_bn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle " + conv_type() + "BN fuse";

    // conv, batch_norm,
    // conv_weight, conv_out,
    // bn_scale, bn_bias, bn_mean, bn_variance,
    // bn_out, bn_mean_out, bn_variance_out, bn_saved_mean,
    // bn_saved_variance
    GET_CONV_BN_NODES(conv_bn_pattern);

    LOG(INFO) << conv->Name();
    LOG(INFO) << batch_norm->Name();
    LOG(INFO) << conv_weight->Name();
    LOG(INFO) << conv_out->Name();
    LOG(INFO) << bn_scale->Name();
    LOG(INFO) << bn_bias->Name();
    LOG(INFO) << bn_mean->Name();
    LOG(INFO) << bn_variance->Name();
    LOG(INFO) << bn_out->Name();
    LOG(INFO) << bn_mean_out->Name();
    LOG(INFO) << bn_variance_out->Name();
    LOG(INFO) << bn_saved_mean->Name();
    LOG(INFO) << bn_saved_variance->Name();

    // check if fuse can be done and if MKL-DNN should be used
    FuseOptions fuse_option = FindFuseOption(*conv, *batch_norm);
    if (fuse_option == DO_NOT_FUSE) {
      VLOG(3) << "do not perform " + conv_type() + " bn fuse";
      return;
    }

    // Get batch norm bias
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();
    LOG(INFO) << "copy bn_bias_tensor";
    copy_tensor_to_cpu(bn_bias_tensor);

    // Create eltwise_y (conv bias) variable
    VarDesc eltwise_y_in_desc(
        patterns::PDNodeName(name_scope_, "eltwise_y_in"));
    eltwise_y_in_desc.SetShape(framework::vectorize(bn_bias_tensor->dims()));
    eltwise_y_in_desc.SetDataType(bn_bias_tensor->type());
    eltwise_y_in_desc.SetLoDLevel(bn_bias->Var()->GetLoDLevel());
    eltwise_y_in_desc.SetPersistable(true);
    auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);
    auto* eltwise_y_in_tensor =
        scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();

    // Initialize eltwise_y
    eltwise_y_in_tensor->Resize(bn_bias_tensor->dims());
    std::fill_n(eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
                eltwise_y_in_tensor->numel(), 0.0f);

    // update weights and biases
    float epsilon =
        BOOST_GET_CONST(float, batch_norm->Op()->GetAttr("epsilon"));
    recompute_bias_and_weights(scope, conv_weight, *bn_scale, *bn_bias_tensor,
                               *bn_mean, *bn_variance, eltwise_y_in_tensor,
                               epsilon, conv_type());

    // with MKL-DNN fuse conv+bn into conv with bias
    // without MKL-DNN fuse conv+bn into conv+elementwise_add
    if (fuse_option == FUSE_NATIVE) {
      // create an elementwise add node.
      OpDesc desc;
      desc.SetInput("X", std::vector<std::string>({conv_out->Name()}));
      desc.SetInput("Y", std::vector<std::string>({eltwise_y_in_node->Name()}));
      desc.SetOutput("Out", std::vector<std::string>({bn_out->Name()}));
      desc.SetType("elementwise_add");
      desc.SetAttr("axis", 1);
      int op_role = BOOST_GET_CONST(
          int, conv->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
      LOG(INFO) << op_role;
      desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
      auto eltwise_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

      GraphSafeRemoveNodes(graph, {batch_norm, bn_mean, bn_variance,
                                   bn_mean_out, bn_variance_out});

      IR_NODE_LINK_TO(conv_out, eltwise_op);
      IR_NODE_LINK_TO(eltwise_y_in_node, eltwise_op);
      IR_NODE_LINK_TO(eltwise_op, bn_out);
      found_conv_bn_count++;
    }
  };

  gpd(graph, handler);

  AddStatis(found_conv_bn_count);
  return graph;
}

// The backward of bn(conv(x))
// op: bn_grad + conv_grad
// bn_grad inputs: x, y_grad, scale, bias, save_mean, save_variance
// bn_grad outputs: x_grad, scale_grad, bias_grad
// conv_grad inputs: input, filter, bias, output_grad
// conv_grad outputs: input_grad(optional), filter_grad
// the output_grad of conv_grad input is the x_grad of bn_grad output
ir::Graph* FuseConvBNPass::FuseConvBNGrad(ir::Graph* graph,
                                          const std::string& opt_type,
                                          bool conv_has_input_grad) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument(
                 "The input graph of FuseConvBNGrad should not be nullptr."));
  FusePassBase::Init("fuse_conv_bn_grad", graph);

  GraphPatternDetector gpd;
  patterns::FuseConvBNGrad fuse_conv_bn_grad_pattern(gpd.mutable_pattern(),
                                                     "fuse_conv_bn_grad");
  fuse_conv_bn_grad_pattern(conv_type() + "_grad", opt_type,
                            conv_has_input_grad);
  int found_conv_bn_grad_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle FuseConvBNGrad";

#define SIMPLE_GET_IR_NODE(var, pat) GET_IR_NODE_FROM_SUBGRAPH(var, var, pat);

    SIMPLE_GET_IR_NODE(bn_grad, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(conv_grad, fuse_conv_bn_grad_pattern);

    SIMPLE_GET_IR_NODE(bn_in_x, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_y_grad, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_scale, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_bias, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_save_mean, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_save_variance, fuse_conv_bn_grad_pattern);

    SIMPLE_GET_IR_NODE(bn_out_x_grad, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_out_scale_grad, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_out_bias_grad, fuse_conv_bn_grad_pattern);

    SIMPLE_GET_IR_NODE(conv_in_input, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(conv_in_filter, fuse_conv_bn_grad_pattern);
    // conv_in_output_grad is bn_out_x_grad

    Node* conv_out_input_grad_node = nullptr;
    if (conv_has_input_grad) {
      SIMPLE_GET_IR_NODE(conv_out_input_grad, fuse_conv_bn_grad_pattern);
      conv_out_input_grad_node = conv_out_input_grad;
    }
    SIMPLE_GET_IR_NODE(conv_out_filter_grad, fuse_conv_bn_grad_pattern);

    SIMPLE_GET_IR_NODE(bn_in_scale_opt, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_scale_opt_out, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_bias_opt, fuse_conv_bn_grad_pattern);
    SIMPLE_GET_IR_NODE(bn_in_bias_opt_out, fuse_conv_bn_grad_pattern);
#undef SIMPLE_GET_IR_NODE

    OpDesc desc;
    desc.SetType(conv_type() + "_grad");
    desc.SetInput("Input", {conv_in_input->Name()});
    desc.SetInput("Filter", {conv_in_filter->Name()});
    desc.SetInput(GradVarName("Output"), {bn_in_y_grad->Name()});
    if (conv_has_input_grad) {
      desc.SetOutput(GradVarName("Input"), {conv_out_input_grad_node->Name()});
    }
    desc.SetOutput(GradVarName("Filter"), {conv_out_filter_grad->Name()});
    for (auto& m : conv_grad->Op()->GetAttrMap()) {
      desc.SetAttr(m.first, m.second);
    }
    auto fused_node = g->CreateOpNode(&desc);

    IR_NODE_LINK_TO(conv_in_input, fused_node);
    IR_NODE_LINK_TO(conv_in_filter, fused_node);
    IR_NODE_LINK_TO(bn_in_y_grad, fused_node);
    if (conv_has_input_grad) {
      IR_NODE_LINK_TO(fused_node, conv_out_input_grad_node);
    }
    IR_NODE_LINK_TO(fused_node, conv_out_filter_grad);

    GraphSafeRemoveNodes(
        g, {bn_grad, conv_grad, bn_in_scale, bn_in_bias, bn_in_save_mean,
            bn_in_save_variance, bn_out_x_grad, bn_out_scale_grad,
            bn_out_bias_grad, bn_in_scale_opt, bn_in_scale_opt_out,
            bn_in_bias_opt, bn_in_bias_opt_out});
    found_conv_bn_grad_count++;
  };

  gpd(graph, handler);
  AddStatis(found_conv_bn_grad_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_conv_bn_pass, paddle::framework::ir::FuseConvBNPass);
REGISTER_PASS_CAPABILITY(fuse_conv_bn_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("batch_norm", 0));

REGISTER_PASS(fuse_depthwise_conv_bn_pass,
              paddle::framework::ir::FuseDepthwiseConvBNPass);
