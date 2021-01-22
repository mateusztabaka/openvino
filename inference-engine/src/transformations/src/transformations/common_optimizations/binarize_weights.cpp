// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/binarize_weights.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(pass::BinarizeWeights, "BinarizeWeights", 0);

// Perform weights quantization using following steps
// if (weights <= input_low)
//    return output_low;
// if (weights > input_high)
//    return output_high;
// return round((f - input_low) / (input_high - input_low)) * (output_high - output_low) + output_low;
std::shared_ptr<Node> quantize_weights(const Output<Node>& weights, const Output<Node>& input_low, const Output<Node>& input_high,
                                       const Output<Node>& output_low, const Output<Node>& output_high) {
    // if (weights <= input_low)
    //    return output_low;
    auto less_eq = std::make_shared<opset5::Convert>(std::make_shared<opset5::LessEqual>(weights, input_low), element::f32);
    auto output_low_branch = std::make_shared<opset5::Multiply>(less_eq, output_low);
    // if (weights > input_high)
    //    return output_high;
    auto greater = std::make_shared<opset5::Convert>(std::make_shared<opset5::Greater>(weights, input_high), element::f32);
    auto output_high_branch = std::make_shared<opset5::Multiply>(greater, output_high);
    // round((f - input_low) / (input_high - input_low)) * (output_high - output_low) + output_low;
    auto round_branch = std::make_shared<opset5::Add>(
        std::make_shared<opset5::Multiply>(
            std::make_shared<opset5::Round>(
                std::make_shared<opset5::Divide>(
                    std::make_shared<opset5::Subtract>(weights, input_low),
                    std::make_shared<opset5::Subtract>(input_high, input_low)),
                opset5::Round::RoundMode::HALF_AWAY_FROM_ZERO),
            std::make_shared<opset5::Subtract>(output_high, output_low)),
        output_low);
    // output_low_branch + output_high_branch + round_branch
    return std::make_shared<opset5::Add>(std::make_shared<opset5::Add>(output_low_branch, output_high_branch), round_branch);
}

// Perform weights quantization using following steps
// if (weights <= input_thr)
//    return output_low;
// if (weights > input_thr)
//    return output_high;
std::shared_ptr<Node> quantize_weights(const Output<Node>& weights, const Output<Node>& input_thr,
                                       const Output<Node>& output_low, const Output<Node>& output_high) {
    // if (weights <= input_thr)
    //    return output_low;
    auto less_eq = std::make_shared<opset5::Convert>(std::make_shared<opset5::LessEqual>(weights, input_thr), element::f32);
    auto output_low_branch = std::make_shared<opset5::Multiply>(less_eq, output_low);
    // if (weights > input_thr)
    //    return output_high;
    auto greater = std::make_shared<opset5::Convert>(std::make_shared<opset5::Greater>(weights, input_thr), element::f32);
    auto output_high_branch = std::make_shared<opset5::Multiply>(greater, output_high);
    // output_low_branch + output_high_branch
    return std::make_shared<opset5::Add>(output_low_branch, output_high_branch);
}



pass::BinarizeWeights::BinarizeWeights() {
    MATCHER_SCOPE(BinarizeWeights);
    auto activations_fq_pattern = pattern::wrap_type<opset5::FakeQuantize>(
            {pattern::any_input(),
             pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>()},
            pattern::consumers_count(1));
    auto weights_fq_pattern = pattern::wrap_type<opset5::FakeQuantize>(
            {pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>(),
             pattern::wrap_type<opset5::Constant>()},
            pattern::consumers_count(1));
    auto conv_pattern = pattern::wrap_type<opset5::Convolution>({activations_fq_pattern, weights_fq_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher &m) {
        auto conv = std::dynamic_pointer_cast<opset5::Convolution>(m.get_match_root());
        if (!conv)
            return false;
        auto activations_fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(conv->input_value(0).get_node_shared_ptr());
        if (!activations_fq || activations_fq->get_levels() != 2)
            return false;
        auto weights_fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(conv->input_value(1).get_node_shared_ptr());
        if (!weights_fq || weights_fq->get_levels() != 2)
            return false;

        auto weights_const = std::dynamic_pointer_cast<opset5::Constant>(weights_fq->input_value(0).get_node_shared_ptr());
        if (!weights_const)
            return false;

        auto check_output_low_high = [] (const std::shared_ptr<opset5::Constant>& output_low_const,
                                         const std::shared_ptr<opset5::Constant>& output_high_const) -> std::tuple<bool, bool, bool> {
            auto output_low = output_low_const->cast_vector<float>();
            auto output_high = output_high_const->cast_vector<float>();
            bool output_low_is_zero = true;
            bool output_high_is_zero = true;
            bool output_low_high_are_opposite = true;
            for (size_t i = 0; i < output_low.size(); i++) {
                output_low_is_zero = output_low_is_zero && output_low[i] == 0.0f;
                output_high_is_zero = output_high_is_zero && output_high[i] == 0.0f;
                output_low_high_are_opposite = output_low_high_are_opposite && output_low[i] == -output_high[i];
            }
            return std::tuple<bool, bool, bool>{output_low_is_zero, output_high_is_zero, output_low_high_are_opposite};
        };

        auto activations_output_low_const = std::dynamic_pointer_cast<opset5::Constant>(activations_fq->input_value(3).get_node_shared_ptr());
        auto activations_output_high_const = std::dynamic_pointer_cast<opset5::Constant>(activations_fq->input_value(4).get_node_shared_ptr());
        if (!activations_output_low_const || !activations_output_high_const)
            return false;
        bool act_out_low_is_zero = false;
        bool act_out_high_is_zero = false;
        bool act_out_low_high_are_opposite = false;
        std::tie(act_out_low_is_zero, act_out_high_is_zero, act_out_low_high_are_opposite) = check_output_low_high(activations_output_low_const,
                                                                                                                   activations_output_high_const);
        if (!(act_out_low_high_are_opposite || (act_out_low_is_zero ^ act_out_high_is_zero)))
            return false;

        auto weights_input_low_const = std::dynamic_pointer_cast<opset5::Constant>(weights_fq->input_value(1).get_node_shared_ptr());
        auto weights_input_high_const = std::dynamic_pointer_cast<opset5::Constant>(weights_fq->input_value(2).get_node_shared_ptr());
        if (!weights_input_low_const || !weights_input_high_const)
            return false;
        auto weights_output_low_const = std::dynamic_pointer_cast<opset5::Constant>(weights_fq->input_value(3).get_node_shared_ptr());
        auto weights_output_high_const = std::dynamic_pointer_cast<opset5::Constant>(weights_fq->input_value(4).get_node_shared_ptr());
        if (!weights_output_low_const || !weights_output_high_const)
            return false;
        bool weights_out_low_is_zero = false;
        bool weights_out_high_is_zero = false;
        bool weights_out_low_high_are_opposite = false;
        std::tie(weights_out_low_is_zero, weights_out_high_is_zero, weights_out_low_high_are_opposite) = check_output_low_high(weights_output_low_const,
                                                                                                                               weights_output_high_const);
        if (!(weights_out_low_high_are_opposite || (weights_out_low_is_zero ^ weights_out_high_is_zero)))
            return false;

        auto weights_input_low = weights_input_low_const->cast_vector<float>();
        auto weights_input_high = weights_input_high_const->cast_vector<float>();
        bool weights_in_low_and_high_are_equal = true;
        for (size_t i = 0; i < weights_input_low.size(); i++) {
            weights_in_low_and_high_are_equal = weights_in_low_and_high_are_equal &&
                std::fabs(weights_input_low[i] - weights_input_high[i]) < std::numeric_limits<float>::epsilon();
        }

        std::shared_ptr<Node> activations_norm_factor;
        if (act_out_high_is_zero)
            activations_norm_factor = activations_output_low_const;
        else
            activations_norm_factor = activations_output_high_const;
        std::shared_ptr<Node> weights_norm_factor;
        if (weights_out_high_is_zero)
            weights_norm_factor = weights_output_low_const;
        else
            weights_norm_factor = weights_output_high_const;

        auto output_low_normalized = std::make_shared<opset5::Divide>(activations_output_low_const, activations_norm_factor);
        output_low_normalized->set_friendly_name(activations_output_low_const->get_friendly_name());
        auto output_high_normalized = std::make_shared<opset5::Divide>(activations_output_high_const, activations_norm_factor);
        output_high_normalized->set_friendly_name(activations_output_high_const->get_friendly_name());
        auto new_activations_fq = activations_fq->clone_with_new_inputs({activations_fq->input_value(0),
                                                                         activations_fq->input_value(1),
                                                                         activations_fq->input_value(2),
                                                                         output_low_normalized,
                                                                         output_high_normalized});
        new_activations_fq->set_friendly_name(activations_fq->get_friendly_name());

        std::shared_ptr<Node> quantized_weights;
        if (weights_in_low_and_high_are_equal) {
            quantized_weights = quantize_weights(weights_const, weights_input_low_const,
                                                 std::make_shared<opset5::Divide>(weights_output_low_const, weights_norm_factor),
                                                 std::make_shared<opset5::Divide>(weights_output_high_const, weights_norm_factor));
        } else {
            quantized_weights = quantize_weights(weights_const, weights_input_low_const, weights_input_high_const,
                                                 std::make_shared<opset5::Divide>(weights_output_low_const, weights_norm_factor),
                                                 std::make_shared<opset5::Divide>(weights_output_high_const, weights_norm_factor));
        }
        quantized_weights->set_friendly_name(weights_fq->get_friendly_name());
        auto new_conv = conv->clone_with_new_inputs({new_activations_fq, quantized_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());

        std::vector<int64_t> norm_factor_shape = {-1};
        for (size_t i = 2; i < weights_const->get_shape().size(); i++)
            norm_factor_shape.push_back(1);
        auto norm_factor_shape_const = opset5::Constant::create(element::i64, Shape{norm_factor_shape.size()}, norm_factor_shape);

        auto activations_norm_factor_reshaped = std::make_shared<opset5::Reshape>(activations_norm_factor, norm_factor_shape_const, false);
        auto mul = std::make_shared<opset5::Multiply>(new_conv, activations_norm_factor_reshaped);
        auto weights_norm_factor_reshaped = std::make_shared<opset5::Reshape>(weights_norm_factor, norm_factor_shape_const, false);
        auto mul2 = std::make_shared<opset5::Multiply>(mul, weights_norm_factor_reshaped);

        replace_node(conv, mul2);
        copy_runtime_info({activations_fq, conv}, {new_activations_fq, new_conv, mul, mul2});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(conv_pattern, matcher_name);
    this->register_matcher(m, callback);
}
