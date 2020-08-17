// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_weightsquantize.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <cassert>

ngraph::pass::ConvertWeightsQuantize::ConvertWeightsQuantize() {
    //auto m_w_const = ngraph::pattern::wrap_type<opset1::Constant>();
    auto m_w_const = ngraph::pattern::any_input();
    auto m_w_convert = ngraph::pattern::wrap_type<opset1::Convert>({m_w_const});
    auto m_w_zero_point = ngraph::pattern::wrap_type<opset1::Constant>();
    auto m_sub = ngraph::pattern::wrap_type<opset1::Subtract>({m_w_convert, m_w_zero_point}, pattern::consumers_count(1));
    auto m_w_scale = ngraph::pattern::wrap_type<opset1::Constant>();
    auto m_mul = ngraph::pattern::wrap_type<opset1::Multiply>({m_sub, m_w_scale});

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        auto mul = std::dynamic_pointer_cast<ngraph::opset1::Multiply> (m.get_match_root());
        if (!mul)
            return false;

        auto pattern_map = m.get_pattern_map();
        auto w_convert = pattern_map[m_w_const];
        float min = 0;
        float max = 0;
        switch (w_convert->get_element_type()) {
        case element::Type_t::i8:
            min = static_cast<float>(std::numeric_limits<int8_t>::min());
            max = static_cast<float>(std::numeric_limits<int8_t>::max());
            break;
        case element::Type_t::u8:
            min = static_cast<float>(std::numeric_limits<uint8_t>::min());
            max = static_cast<float>(std::numeric_limits<uint8_t>::max());
            break;
        }
        size_t levels = max - min + 1;
        auto input_low = opset1::Constant::create(element::f32, Shape{}, {min});
        auto input_high = opset1::Constant::create(element::f32, Shape{}, {max});
        auto levels_minus_one = opset1::Constant::create(element::f32, Shape{}, {static_cast<float>(levels - 1)});
        auto zero_point = pattern_map[m_w_zero_point];
        auto scale = pattern_map[m_w_scale];

        // out_low = - zeropoint * scale
        auto out_low = std::make_shared<ngraph::opset1::Negative>(std::make_shared<ngraph::opset1::Multiply>(scale, zero_point));
        // out_high = scale * (levels - 1) - zeropoint * scale
        auto out_high = std::make_shared<ngraph::opset1::Add>(
                std::make_shared<ngraph::opset1::Multiply>(scale, levels_minus_one), out_low);
        auto fake_q = std::make_shared<ngraph::opset1::FakeQuantize>(w_convert, input_low, input_high, out_low, out_high, levels);

        fake_q->set_friendly_name(mul->get_friendly_name());

        copy_runtime_info(mul, {w_convert, input_low, input_high, levels_minus_one, zero_point, scale, out_low, out_high, fake_q});
        replace_node(mul, fake_q);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_mul, "ConvertWeightsQuantize");
    this->register_matcher(m, callback);
}
