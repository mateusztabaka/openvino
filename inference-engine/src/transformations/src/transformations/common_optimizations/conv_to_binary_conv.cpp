// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/conv_to_binary_conv.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvToBinaryConv, "ConvToBinaryConv", 0);

ngraph::pass::ConvToBinaryConv::ConvToBinaryConv() {
    auto fq_act_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>(
            {ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input()});
    auto fq_weights_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>(
            {ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>()});
    auto conv_pattern = ngraph::pattern::wrap_type<opset5::Convolution>({fq_act_pattern, fq_weights_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto conv = std::dynamic_pointer_cast<opset5::Convolution>(m.get_match_root());
        if (!conv)
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto fq_act = std::dynamic_pointer_cast<opset5::FakeQuantize>(conv->input_value(0).get_node_shared_ptr());
        if (!fq_act || fq_act->get_levels() != 2)
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto fq_weights = std::dynamic_pointer_cast<opset5::FakeQuantize>(conv->input_value(1).get_node_shared_ptr());
        if (!fq_weights || fq_weights->get_levels() != 2)
            return false;
        auto input_low_constant = std::dynamic_pointer_cast<opset5::Constant>(fq_weights->input_value(3).get_node_shared_ptr());
        if (!input_low_constant)
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto input_high_constant = std::dynamic_pointer_cast<opset5::Constant>(fq_weights->input_value(4).get_node_shared_ptr());
        if (!input_high_constant)
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto output_low_constant = std::dynamic_pointer_cast<opset5::Constant>(fq_weights->input_value(3).get_node_shared_ptr());
        if (!output_low_constant)
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto output_high_constant = std::dynamic_pointer_cast<opset5::Constant>(fq_weights->input_value(4).get_node_shared_ptr());
        if (!output_high_constant)
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";
        auto input_low = input_low_constant->cast_vector<float>()[0];
        auto input_high = input_high_constant->cast_vector<float>()[0];
        auto output_low = output_low_constant->cast_vector<float>()[0];
        auto output_high = output_high_constant->cast_vector<float>()[0];
        if (!(output_low == -output_high || ((output_low == 0.0f) ^ (output_high == 0.0f))))
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";

        float norm_factor = output_high == 0.0f ? output_low : output_high;
        auto norm_factor_constant = opset5::Constant::create(element::f32, Shape{}, {norm_factor});
        output_low = std::roundf(output_low / norm_factor);
        output_high = std::roundf(output_high / norm_factor);

        auto fq_input_constant = std::dynamic_pointer_cast<opset5::Constant>(fq_weights->input_value(0).get_node_shared_ptr());
        if (!fq_input_constant)
            return false;
        auto fq_input = fq_input_constant->cast_vector<float>();
        std::vector<float> weights;
        weights.resize(fq_input.size());
        std::cout << "input low " << input_low << " input high " << input_high << " out low " << output_low << " output high " << output_high << std::endl;
        std::transform(fq_input.begin(), fq_input.end(), std::back_inserter(weights), [input_low, input_high, output_low, output_high] (float f) -> float {
            std::cout << "f " << f << std::endl;
            if (f <= input_low) {
                return output_low;
            } else if (f > input_high) {
                return output_high;
            } else {
                return std::roundf((f - input_low) / (input_high - input_low)) * (output_high - output_low) + output_low;
            }
        });
        std::cout << "WEIGHTS\n";
        for (auto w : weights)
            std::cout << w << "\n";
        std::cout << std::endl;
        if (!std::all_of(weights.begin(), weights.end(), [] (float f) -> bool { return f == 1.0f || f == -1.0f; }))
            return false;
        std::cout << __FILE__ << ":" << __LINE__ << "!!!!!!!!!!!!!!!!!!!!Conv to bin conv\n";

        if (output_low == 0.0f) {
            auto new_conv = std::make_shared<opset5::BinaryConvolution>(conv->input_value(0), conv->input_value(1),
                                                                        conv->get_strides(),
                                                                        conv->get_pads_begin(),
                                                                        conv->get_pads_end(),
                                                                        conv->get_dilations(),
                                                                        opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                                                        -1,
                                                                        conv->get_auto_pad());
            new_conv->set_friendly_name(conv->get_friendly_name());
            std::vector<int64_t> axes;
            for (size_t i = 1; i < fq_input_constant->get_shape().size(); i++)
                axes.push_back(i);
            auto weights_reduced = std::make_shared<opset5::ReduceSum>(op::Constant::create(element::f32, fq_input_constant->get_shape(), weights),
                    op::Constant::create(element::i64, Shape{axes.size()}, axes), true);
            auto add = std::make_shared<opset5::Add>(new_conv, weights_reduced);
            auto mul = std::make_shared<opset5::Multiply>(add, op::Constant::create(element::f32, Shape{}, {0.5}));
            auto mul2 = std::make_shared<opset5::Multiply>(mul, norm_factor_constant);
            replace_node(conv, mul2);
            copy_runtime_info(conv, {new_conv, add, mul, mul2});

            return true;
        }

        auto new_conv = std::make_shared<opset5::BinaryConvolution>(conv->input_value(0), conv->input_value(1),
                                                                    conv->get_strides(),
                                                                    conv->get_pads_begin(),
                                                                    conv->get_pads_end(),
                                                                    conv->get_dilations(),
                                                                    opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                                                    0,
                                                                    conv->get_auto_pad());
        new_conv->set_friendly_name(conv->get_friendly_name());
        auto mul = std::make_shared<opset5::Multiply>(new_conv, norm_factor_constant);
        replace_node(conv, mul);
        copy_runtime_info(conv, {new_conv, mul});

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, "ConvToBinaryConv");
    this->register_matcher(m, callback);
}
