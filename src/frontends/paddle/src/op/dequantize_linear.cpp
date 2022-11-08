// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs dequantize_linear(const NodeContext& node) {
    // extract the INPUTS
    const auto x = node.get_input("X");
    const auto scale = node.get_input("Scale"); // type: float or 1-D
    const auto zero_point = node.get_input("ZeroPoint");

    // assert shape of scale and zero_point
    const auto& scale_shape = scale.get_partial_shape();

    if (scale_shape.rank().get_length() == 1) {
        PADDLE_OP_CHECK(node, scale.get_partial_shape() == zero_point.get_partial_shape(), "dequantize_linear shape of scale and zero_point doesn't match.");
    } else if (scale_shape.rank().get_length() == 2) {
        PADDLE_OP_CHECK(node, scale.get_partial_shape()[1] == zero_point.get_partial_shape()[0], "dequantize_linear shape of scale and zero_point doesn't match.");
    } else {
        PADDLE_OP_CHECK(node, false, "dims of scale should not be greater than 2.");
    }

    const auto q_node = std::make_shared<default_opset::Convert>(x, element::f32);

    const auto real_scale = std::make_shared<default_opset::Squeeze>(scale);
    
    // extract the ATTRIBUTES
    const auto quant_axis = node.get_attribute<int32_t>("quant_axis");

    //   / == 1-D not need to unsqueeze for broadcast
    // X
    //   \ >= 2-D need to unsqeeze for broadcast
    if (quant_axis == -1) {
        const auto out_node = std::make_shared<default_opset::Multiply>(q_node, real_scale);
        return node.default_single_output_mapping({out_node}, {"Y"});
    } else {
        std::vector<size_t> unsqueeze_pattern(x.get_partial_shape().rank().get_length());
        std::iota(unsqueeze_pattern.begin(), unsqueeze_pattern.end(), 0);
        unsqueeze_pattern.erase(unsqueeze_pattern.begin() + quant_axis);
        const auto unsqueeze_node = std::make_shared<default_opset::Constant>(element::i32, Shape{unsqueeze_pattern.size()}, unsqueeze_pattern);

        const auto scale_ = std::make_shared<default_opset::Unsqueeze>(real_scale, unsqueeze_node);

        const auto out_node = std::make_shared<default_opset::Multiply>(q_node, scale_);
        return node.default_single_output_mapping({out_node}, {"Y"});
    } 
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
