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
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>(
            {ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input(),
             ngraph::pattern::any_input()},
            pattern::consumers_count(1));
    auto conv_pattern = ngraph::pattern::wrap_type<opset5::Convolution>({fq_pattern, ngraph::pattern::wrap_type<opset5::Constant>()});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
        std::cout << "Conv to bin conv\n";
        auto conv = std::dynamic_pointer_cast<opset5::Convolution>(m.get_match_root());
        if (!conv)
            return false;
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(conv->input_value(0).get_node_shared_ptr());
        if (!fq || fq->get_levels() != 2)
            return false;
        auto weights_constant = std::dynamic_pointer_cast<opset5::Constant>(conv->input_value(1).get_node_shared_ptr());
        if (!weights_constant)
            return false;
        auto weights = weights_constant->cast_vector<float>();
        if (!std::all_of(weights.begin(), weights.end(), [] (float f) -> bool { return f == 1.0f || f == -1.0f; }))
            return false;

        auto output_low_constant = std::dynamic_pointer_cast<opset5::Constant>(fq->input_value(3).get_node_shared_ptr());
        if (!output_low_constant)
            return false;
        auto output_high_constant = std::dynamic_pointer_cast<opset5::Constant>(fq->input_value(3).get_node_shared_ptr());
        if (!output_high_constant)
            return false;
        auto output_low = output_low_constant->cast_vector<float>();
        auto output_high = output_high_constant->cast_vector<float>();
        bool output_low_is_zero = true;
        bool output_low_is_minus_one = true;
        for (auto f : output_low) {
            output_low_is_zero = output_low_is_zero && f == 0.0f;
            output_low_is_minus_one = output_low_is_minus_one && f == -1.0f;
        }
        if (!output_low_is_zero || !output_low_is_minus_one)
            return false;
        bool output_high_is_one = std::all_of(output_high.begin(), output_high.end(), [] (float f) -> bool { return f == 1.0f; });
        if (!output_high_is_one)
            return false;

        if (output_low_is_zero) {
            std::vector<int64_t> axes;
            for (size_t i = 1; i < weights_constant->get_shape().size(); i++)
                axes.push_back(i);
            auto new_conv = std::make_shared<opset5::BinaryConvolution>(conv->input_value(0), conv->input_value(1),
                                                                        conv->get_strides(),
                                                                        conv->get_pads_begin(),
                                                                        conv->get_pads_end(),
                                                                        conv->get_dilations(),
                                                                        opset5::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT,
                                                                        -1,
                                                                        conv->get_auto_pad());
            new_conv->set_friendly_name(conv->get_friendly_name());
            auto weights_reduced = std::make_shared<opset5::ReduceSum>(weights_constant, op::Constant::create(element::i64, Shape{axes.size()}, axes), true);
            auto add = std::make_shared<opset5::Add>(new_conv, weights_reduced);
            auto mul = std::make_shared<opset5::Multiply>(add, op::Constant::create(element::f32, Shape{}, {0.5}));
            replace_node(conv, mul);
            copy_runtime_info(conv, {new_conv, add, mul});

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
        replace_node(conv, new_conv);
        copy_runtime_info(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, "ConvToBinaryConv");
    this->register_matcher(m, callback);
}
