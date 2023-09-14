// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compress_quantize_weights.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "validation_util.hpp"

static bool has_dequantization_subgraph(const std::shared_ptr<ov::Node>& first_convert);
static std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights(const std::shared_ptr<ov::Node>& quantize,
                                                                        const std::shared_ptr<ov::Node>& convert);
static void replace_with_dequantize_subgraph(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq,
                                             const std::shared_ptr<ov::op::v0::Constant>& new_weights,
                                             const ov::element::Type& high_prec_type,
                                             const std::shared_ptr<ov::op::v0::Constant>& scale,
                                             const std::shared_ptr<ov::op::v0::Constant>& zero_point = nullptr);
static std::shared_ptr<ov::op::v0::Constant> try_to_fuse_zero_point(
    const std::shared_ptr<ov::op::v0::Constant>& weights,
    const std::vector<float>& zero_point_values,
    const ov::element::Type& high_prec_type,
    const ov::element::Type& low_prec_type,
    const ov::Shape& zero_point_shape);

ov::pass::CompressQuantizeWeights::CompressQuantizeWeights() {
    auto weights_const_pattern = pattern::wrap_type<op::v0::Constant>();
    auto weigths_convert_pattern = pattern::wrap_type<opset8::Convert>({weights_const_pattern});
    OutputVector weights_options{weights_const_pattern, weigths_convert_pattern};
    auto weights_pattern = std::make_shared<pattern::op::Or>(weights_options);
    auto input_low_pattern = pattern::wrap_type<op::v0::Constant>();
    auto input_high_pattern = pattern::wrap_type<op::v0::Constant>();
    auto output_low_pattern = pattern::wrap_type<op::v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<op::v0::Constant>();
    auto fq_pattern = pattern::wrap_type<opset8::FakeQuantize>(
        {weights_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = std::dynamic_pointer_cast<opset8::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        auto levels = fq->get_levels();
        if (levels <= 2 || levels > 256)
            return false;
        auto low_prec_type = element::undefined;
        // Currently we support two weights quantize types: i4 and i8
        if (levels <= 16) {
            low_prec_type = element::i4;
        } else if (levels <= 256) {
            low_prec_type = element::i8;
        }

        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& high_prec_type = fq->get_element_type();
        const auto& fq_data_input = fq->get_input_node_shared_ptr(0);

        // skip dequantize part if there is already dequantization subgraph after FakeQuantize
        auto fq_users = fq->get_users();
        if (fq_users.size() == 1 && has_dequantization_subgraph(fq_users[0])) {
            const auto& convert = fq_users[0];
            auto new_weights = compress_quantized_weights(fq, convert);
            if (!new_weights)
                return false;
            new_weights->set_friendly_name(convert->get_friendly_name());
            replace_node(convert, new_weights);
            copy_runtime_info(convert, new_weights);
            // preserve dequantization subgraph for LP transformations
            auto weights_users = new_weights->get_users();
            if (weights_users.size() == 1 && ov::is_type<ov::opset8::Convert>(weights_users[0])) {
                ov::pass::disable_constant_folding(weights_users[0]);
            }
            return true;
        } else {
            /*
               Quantize part

               Prepare new FakeQuantize that performs weights quantization.
               In this case input_low/high stays the same, but we need new output_low/high:
                 output_low = -levels / 2
                 output_high = levels - 1 + output_low
               The FakeQuantize result is converted to low precision type and then constant folded
            */
            float new_output_low_value = -static_cast<float>(levels / 2);
            float new_output_high_value = levels - 1 + new_output_low_value;
            std::shared_ptr<Node> new_output_low =
                op::v0::Constant::create(high_prec_type, Shape{}, {new_output_low_value});
            std::shared_ptr<Node> new_output_high =
                op::v0::Constant::create(high_prec_type, Shape{}, {new_output_high_value});
            const auto& weights_const = pattern_value_map.at(weights_const_pattern);
            Output<Node> input_low = pattern_value_map.at(input_low_pattern);
            Output<Node> input_high = pattern_value_map.at(input_high_pattern);
            auto quantize =
                fq->clone_with_new_inputs({fq_data_input, input_low, input_high, new_output_low, new_output_high});
            auto convert = std::make_shared<opset8::Convert>(quantize, low_prec_type);
            auto new_weights = compress_quantized_weights(quantize, convert);
            if (!new_weights)
                return false;
            new_weights->set_friendly_name(weights_const.get_node()->get_friendly_name());

            /*
               Dequantize part is performed by Convert(from low to high precision)->Subtract->Multiply subgraph.

                                 +-------------------------+
                                 |         Convert         |
                                 | (from low to high prec) |
                                 +-------------------------+
                                              |
                                              v
                        +----------+    +------------+
                        |zero point|--->|  Subtract  |
                        +----------+    +-----+------+
                                              |
                                              v
                         +---------+    +------------+
                         |  scale  |--->|  Multiply  |
                         +---------+    +-----+------+
                                              |
                                              v

                where:
                    scale = (output_high - output_low) / (new_output_high - new_output_low)
                    zero_point = new_output_low - output_low / scale
            */
            Output<Node> output_low = pattern_value_map.at(output_low_pattern);
            Output<Node> output_high = pattern_value_map.at(output_high_pattern);
            const auto& fq_type = fq->get_output_element_type(0);
            const bool should_convert_intervals = fq_type.is_real() && fq_type.size() < element::f32.size();
            if (should_convert_intervals) {
                input_low = std::make_shared<opset8::Convert>(input_low, element::f32);
                input_high = std::make_shared<opset8::Convert>(input_high, element::f32);
                output_low = std::make_shared<opset8::Convert>(output_low, element::f32);
                output_high = std::make_shared<opset8::Convert>(output_high, element::f32);
                new_output_low = std::make_shared<opset8::Convert>(new_output_low, element::f32);
                new_output_high = std::make_shared<opset8::Convert>(new_output_high, element::f32);
            }
            auto output_range = std::make_shared<opset8::Subtract>(output_high, output_low);
            auto input_range = op::v0::Constant::create(new_output_low->get_output_element_type(0),
                                                        Shape{},
                                                        {new_output_high_value - new_output_low_value});
            auto scale = ov::util::constantfold_subgraph(std::make_shared<opset8::Divide>(output_range, input_range));
            if (!scale)
                return false;
            auto shift = ov::util::constantfold_subgraph(
                std::make_shared<opset8::Subtract>(new_output_low,
                                                   std::make_shared<op::v1::Divide>(output_low, scale)));
            if (!shift)
                return false;

            auto scale_values = scale->cast_vector<float>();
            auto shift_values = shift->cast_vector<float>();
            std::vector<float> zero_point_values;
            zero_point_values.reserve(scale_values.size());

            // shift equals to input_low - output_low / scale
            // for positions where scale == 0, we put zero as shift
            bool zero_point_is_zero = true;
            for (size_t i = 0; i < scale_values.size(); i++) {
                zero_point_values.push_back((scale_values[i] != 0) * shift_values[i]);
                zero_point_is_zero =
                    zero_point_is_zero && (std::fabs(zero_point_values.back()) < std::numeric_limits<float>::epsilon());
            }

            if (should_convert_intervals) {
                scale = ov::util::constantfold_subgraph(std::make_shared<opset8::Convert>(scale, fq_type));
                if (!scale)
                    return false;
            }

            if (zero_point_is_zero) {
                replace_with_dequantize_subgraph(fq, new_weights, high_prec_type, scale);
                return true;
            }

            const Shape& zero_point_shape = scale->get_shape();
            auto new_weights_with_fused_zp =
                try_to_fuse_zero_point(new_weights, zero_point_values, high_prec_type, low_prec_type, zero_point_shape);
            if (new_weights_with_fused_zp) {
                replace_with_dequantize_subgraph(fq, new_weights_with_fused_zp, high_prec_type, scale);
            } else {
                auto zero_point = op::v0::Constant::create(high_prec_type, zero_point_shape, zero_point_values);
                replace_with_dequantize_subgraph(fq, new_weights, high_prec_type, scale, zero_point);
            }

            return true;
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fq_pattern, "CompressQuantizeWeights");
    this->register_matcher(m, callback);
}

ov::Tensor tensor_from_constant(const std::shared_ptr<ov::op::v0::Constant>& constant) {
    return ov::Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr()));
}

bool evaluate_node(const std::shared_ptr<ov::Node>& node,
                   const ov::TensorVector& input_tensors,
                   ov::Tensor& output_tensor) {
    if (node->get_output_size() != 1)
        return false;

    ov::TensorVector output_tensors{ov::Tensor(node->get_output_element_type(0), node->get_output_shape(0))};
    if (!node->evaluate(output_tensors, input_tensors))
        return false;

    output_tensor = output_tensors[0];

    return true;
}

ov::TensorVector get_fake_quantize_input_tensors(const std::shared_ptr<ov::Node>& fq) {
    ov::Tensor weights_tensor;

    auto fq_input = fq->get_input_node_shared_ptr(0);
    auto fq_input_constant = ov::as_type_ptr<ov::op::v0::Constant>(fq_input);

    if (!fq_input_constant) {
        auto weights = ov::as_type_ptr<ov::op::v0::Constant>(fq_input->get_input_node_shared_ptr(0));
        if (!evaluate_node(fq_input, ov::TensorVector{tensor_from_constant(weights)}, weights_tensor))
            return {};
    } else {
        weights_tensor = tensor_from_constant(fq_input_constant);
    }

    auto in_low = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(1));
    auto in_high = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(2));
    auto out_low = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(3));
    auto out_high = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(4));

    return ov::TensorVector{weights_tensor,
                            tensor_from_constant(in_low),
                            tensor_from_constant(in_high),
                            tensor_from_constant(out_low),
                            tensor_from_constant(out_high)};
}

bool has_dequantization_subgraph(const std::shared_ptr<ov::Node>& first_convert) {
    auto first_convert_users = first_convert->get_users();
    const auto second_convert = std::find_if(first_convert_users.begin(),
                                             first_convert_users.end(),
                                             [](const std::shared_ptr<ov::Node>& n) -> bool {
                                                 return ov::is_type<ov::opset8::Convert>(n);
                                             });
    if (second_convert == first_convert_users.end())
        return false;
    auto convert_or_subtract_users = (*second_convert)->get_users();
    const auto subtract = std::find_if(convert_or_subtract_users.begin(),
                                       convert_or_subtract_users.end(),
                                       [](const std::shared_ptr<ov::Node>& n) -> bool {
                                           return ov::is_type<ov::opset8::Subtract>(n);
                                       });
    if (subtract != convert_or_subtract_users.end()) {
        convert_or_subtract_users = (*subtract)->get_users();
    }
    const auto multiply = std::find_if(convert_or_subtract_users.begin(),
                                       convert_or_subtract_users.end(),
                                       [](const std::shared_ptr<ov::Node>& n) -> bool {
                                           return ov::is_type<ov::opset8::Multiply>(n);
                                       });
    return multiply != convert_or_subtract_users.end();
}

std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights(const std::shared_ptr<ov::Node>& quantize,
                                                                 const std::shared_ptr<ov::Node>& convert) {
    ov::Tensor quantize_output_tensor;
    if (!evaluate_node(quantize, get_fake_quantize_input_tensors(quantize), quantize_output_tensor))
        return nullptr;
    ov::Tensor new_weights_tensor;
    if (!evaluate_node(convert, {quantize_output_tensor}, new_weights_tensor))
        return nullptr;
    return std::make_shared<ov::op::v0::Constant>(new_weights_tensor);
}

void replace_with_dequantize_subgraph(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq,
                                      const std::shared_ptr<ov::op::v0::Constant>& new_weights,
                                      const ov::element::Type& high_prec_type,
                                      const std::shared_ptr<ov::op::v0::Constant>& scale,
                                      const std::shared_ptr<ov::op::v0::Constant>& zero_point) {
    ov::pass::NodeRegistry node_registry;
    auto convert = node_registry.make<ov::op::v0::Convert>(new_weights, high_prec_type);
    ov::pass::disable_constant_folding(convert);
    std::shared_ptr<ov::op::v1::Multiply> mul;
    if (zero_point) {
        auto sub = node_registry.make<ov::op::v1::Subtract>(convert, zero_point);
        mul = node_registry.make<ov::op::v1::Multiply>(sub, scale);
    } else {
        mul = node_registry.make<ov::op::v1::Multiply>(convert, scale);
    }
    mul->set_friendly_name(fq->get_friendly_name());
    copy_runtime_info(fq, node_registry.get());
    replace_node(fq, mul);
}

std::shared_ptr<ov::op::v0::Constant> try_to_fuse_zero_point(const std::shared_ptr<ov::op::v0::Constant>& weights,
                                                             const std::vector<float>& zero_point_values,
                                                             const ov::element::Type& high_prec_type,
                                                             const ov::element::Type& low_prec_type,
                                                             const ov::Shape& zero_point_shape) {
    // try to fuse zero point
    std::vector<float> int8_zero_point_values;
    int8_zero_point_values.reserve(zero_point_values.size());
    std::vector<float> adj_zero_point_values;
    adj_zero_point_values.reserve(zero_point_values.size());

    for (size_t i = 0; i < zero_point_values.size(); i++) {
        int8_zero_point_values.push_back(std::nearbyint(zero_point_values[i]));
        adj_zero_point_values.push_back(zero_point_values[i] - int8_zero_point_values.back());
        if (std::fabs(adj_zero_point_values.back()) >= 1e-4) {
            return nullptr;
        }
    }
    auto zero_point = ov::op::v0::Constant::create(high_prec_type, zero_point_shape, zero_point_values);
    auto int8_zero_point = ov::op::v0::Constant::create(low_prec_type, zero_point_shape, int8_zero_point_values);
    auto adj_zero_point = ov::op::v0::Constant::create(high_prec_type, zero_point_shape, adj_zero_point_values);

    auto weights_with_fused_zp =
        ov::util::constantfold_subgraph(std::make_shared<ov::op::v1::Subtract>(weights, int8_zero_point));
    if (!weights_with_fused_zp)
        return nullptr;

    auto transformed = std::make_shared<ov::op::v1::Subtract>(
        std::make_shared<ov::op::v0::Convert>(weights_with_fused_zp, high_prec_type),
        adj_zero_point);
    auto sub_with_zero_point =
        std::make_shared<ov::op::v1::Subtract>(std::make_shared<ov::op::v0::Convert>(weights, high_prec_type),
                                               zero_point);
    auto diff =
        ov::util::constantfold_subgraph(std::make_shared<ov::op::v1::Subtract>(transformed, sub_with_zero_point));
    if (!diff)
        return nullptr;
    auto diff_values = diff->cast_vector<float>();

    if (std::any_of(diff_values.begin(), diff_values.end(), [](float f) {
            return std::fabs(f) >= std::numeric_limits<float>::epsilon();
        }))
        return nullptr;

    return weights_with_fused_zp;
}
