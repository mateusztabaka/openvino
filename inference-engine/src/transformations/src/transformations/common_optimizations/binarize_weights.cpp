// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/binarize_weights.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::BinarizeWeights, "BinarizeWeights", 0);

ngraph::pass::BinarizeWeights::BinarizeWeights() {
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>(
            {ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>(),
             ngraph::pattern::wrap_type<opset5::Constant>()},
            pattern::consumers_count(1));
    auto conv_pattern = ngraph::pattern::wrap_type<opset5::Convolution>({ngraph::pattern::any_input(), fq_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher &m) {
        std::cout << "binarize weights\n";
        auto conv = std::dynamic_pointer_cast<opset5::Convolution>(m.get_match_root());
        if (!conv)
            return false;
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(conv->input_value(1).get_node_shared_ptr());
        if (!fq || fq->get_levels() != 2)
            return false;
        auto weights_constant = std::dynamic_pointer_cast<opset5::Constant>(fq->input_value(0).get_node_shared_ptr());
        if (!weights_constant)
            return false;

        auto output_low_constant = std::dynamic_pointer_cast<opset5::Constant>(fq->input_value(3).get_node_shared_ptr());
        if (!output_low_constant)
            return false;
        auto output_high_constant = std::dynamic_pointer_cast<opset5::Constant>(fq->input_value(4).get_node_shared_ptr());
        if (!output_high_constant)
            return false;

        auto output_low = output_low_constant->cast_vector<float>();
        auto output_high = output_high_constant->cast_vector<float>();
        bool output_low_is_zero = std::all_of(output_low.begin(), output_low.end(), [] (float f) -> bool { return f == 0.0f; });
        bool output_high_is_zero = std::all_of(output_high.begin(), output_high.end(), [] (float f) -> bool { return f == 0.0f; });
        std::vector<bool> opposite;
        opposite.reserve(output_low.size());
        std::transform(output_low.begin(), output_low.end(), output_high.begin(),
                       std::back_inserter(opposite), [] (float f1, float f2) -> bool { return f1 == -f2; });
        bool output_low_and_high_are_opposite = std::all_of(opposite.begin(), opposite.end(), [] (bool b) -> bool { return b; });
        if (!(output_low_and_high_are_opposite || (output_low_is_zero ^ output_high_is_zero)))
            return false;

        std::shared_ptr<Node> norm_factor;
        if (output_high_is_zero)
            norm_factor = fq->input_value(3).get_node_shared_ptr();
        else
            norm_factor = fq->input_value(4).get_node_shared_ptr();

        auto reshape = std::make_shared<opset5::Reshape>(norm_factor, opset5::Constant::create(element::i64, {3}, {-1, 1, 1}), false);
        auto new_output_low = std::make_shared<opset5::Divide>(output_low_constant, reshape);
        auto new_output_high = std::make_shared<opset5::Divide>(output_high_constant, reshape);
        auto new_fq = fq->clone_with_new_inputs({fq->input_value(0), fq->input_value(1), fq->input_value(2), new_output_low, new_output_high});
        auto new_conv = conv->clone_with_new_inputs({conv->input_value(0), new_fq});
        new_conv->set_friendly_name(conv->get_friendly_name());
        auto mul = std::make_shared<opset5::Multiply>(new_conv, reshape);

        replace_node(conv, mul);
        copy_runtime_info({fq, conv}, {reshape, new_output_low, new_output_high, new_fq, new_conv, mul});
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, "BinarizeWeights");
    this->register_matcher(m, callback);
}
