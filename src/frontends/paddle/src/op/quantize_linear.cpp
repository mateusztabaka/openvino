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
/*
    quantize_linear:
    INT ---------------------------> FLOAT
            [scale, zero_point]

    zero_point

                / [-1]      --- per-tensor 
    quant_axis  - [0 or 1]  --- per-channel, expand 1-D tensor to match the input
                \ [others]  --- unsupported     

                / [0]       --- rounding to nearest ties to even
    round_type  - [1]       --- rounding to nearest ties away from zero
                \ [others]  --- unsupported!
    refer to https://en.wikipedia.org/wiki/IEEE_754 for more info about round_type

*/
NamedOutputs quantize_linear(const NodeContext& node) {
    // extract the INPUTS
    const auto x = node.get_input("X");
    const auto scale = node.get_input("Scale"); // type: float or 1-D
    const auto zero_point = node.get_input("ZeroPoint");
    // const auto scale_numel = scale.get_shape().size() == 2 ? scale.get_shape()[1] : scale.get_shape()[0];
    
    // extract the ATTRIBUTES
    const auto bit_length = node.get_attribute<int32_t>("bit_length");
    const auto levels = 1 << bit_length;

    const auto range = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, 127);
    const auto y_scale = std::make_shared<default_opset::Divide>(
                            std::make_shared<default_opset::Squeeze>(scale), range);

    // output
    const auto output_low = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, -128);
    const auto output_high = std::make_shared<default_opset::Constant>(element::f32, Shape{1}, 127);

    // input
    std::shared_ptr<Node> input_low = std::make_shared<default_opset::Multiply>(y_scale, output_low);
    if (auto constant = get_constant_from_source(input_low))
        input_low = constant;
    std::shared_ptr<Node> input_high = std::make_shared<default_opset::Multiply>(y_scale, output_high);
    if (auto constant = get_constant_from_source(input_high))
        input_high = constant;

    const auto q_node = std::make_shared<default_opset::FakeQuantize>(x, input_low, input_high, output_low, output_high, levels);
    const auto out_node = std::make_shared<default_opset::Convert>(q_node, element::i8);
                            
    return node.default_single_output_mapping({out_node}, {"Y"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

