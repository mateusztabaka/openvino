// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_fake_quantize_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include "itt.hpp"


NGRAPH_RTTI_DEFINITION(ngraph::pass::MulFakeQuantizeFusion, "MulFakeQuantizeFusion", 0);

ngraph::pass::MulFakeQuantizeFusion::MulFakeQuantizeFusion() {
    MATCHER_SCOPE(MulFakeQuantizeFusion);
    auto mul_pattern = ngraph::pattern::wrap_type<opset5::Multiply>({ngraph::pattern::any_input(), ngraph::pattern::wrap_type<opset5::Constant>()});
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>({mul_pattern,
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input(),
                                                                        ngraph::pattern::any_input()});
    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        auto mul = std::dynamic_pointer_cast<opset5::Multiply>(fq->input_value(0).get_node_shared_ptr());
        if (!mul)
            return false;
        std::shared_ptr<Node> mul_data = mul->input_value(0).get_node_shared_ptr();
        auto mul_const = std::dynamic_pointer_cast<opset5::Constant>(mul->input_value(1).get_node_shared_ptr());
        if (!mul_const) {
            mul_const = std::dynamic_pointer_cast<opset5::Constant>(mul->input_value(0).get_node_shared_ptr());
            if (!mul_const)
                return false;
            mul_data = mul->input_value(1).get_node_shared_ptr();
        }
        auto mul_const_value = mul_const->cast_vector<float>();
        auto new_input_low = std::make_shared<opset5::Divide>(fq->input_value(1), mul_const);
        auto new_input_high = std::make_shared<opset5::Divide>(fq->input_value(2), mul_const);
        auto new_output_low = fq->input_value(3).get_node_shared_ptr();
        auto new_output_high = fq->input_value(4).get_node_shared_ptr();
        if (std::all_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; })) {
            new_output_low = fq->input_value(4).get_node_shared_ptr();
            new_output_high = fq->input_value(3).get_node_shared_ptr();
        } else if (std::any_of(mul_const_value.begin(), mul_const_value.end(), [] (float f) -> bool { return f < 0.0f; })) {
            auto zero = op::Constant::create(element::f32, Shape{}, {0.0f});
            auto minus_one = op::Constant::create(element::f32, Shape{}, {-1.0f});
            auto less_than_zero = std::make_shared<opset5::Convert>(std::make_shared<opset5::Less>(mul_const, zero), element::f32);
            auto greater_eq_zero = std::make_shared<opset5::Convert>(std::make_shared<opset5::GreaterEqual>(mul_const, zero), element::f32);
            new_output_low = std::make_shared<opset5::Add>(
                    std::make_shared<opset5::Multiply>(std::make_shared<opset5::Multiply>(minus_one, less_than_zero), fq->input_value(3)),
                    std::make_shared<opset5::Multiply>(greater_eq_zero, fq->input_value(3)));
            new_output_high = std::make_shared<opset5::Add>(
                    std::make_shared<opset5::Multiply>(std::make_shared<opset5::Multiply>(minus_one, less_than_zero), fq->input_value(4)),
                    std::make_shared<opset5::Multiply>(greater_eq_zero, fq->input_value(4)));
        }

        auto new_fq = register_new_node<opset5::FakeQuantize>(mul_data, new_input_low,
                new_input_high, fq->input_value(3), fq->input_value(4), fq->get_levels());
        new_fq->set_friendly_name(fq->get_friendly_name());
        copy_runtime_info({mul, fq}, new_fq);
        replace_node(fq, new_fq);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
