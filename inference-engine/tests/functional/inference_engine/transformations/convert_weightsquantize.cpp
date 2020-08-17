// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_weightsquantize.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;

TEST(TransformationTests, ConvertWeightsQuantize) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{3, 1, 2}, {0, 1, 2, 3, 4, 5});
        auto zero_point = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2});
        auto scale = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {5});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(
                std::make_shared<ngraph::opset1::Subtract>(std::make_shared<ngraph::opset1::Convert>(data, ngraph::element::f32), zero_point),
                scale);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertWeightsQuantize>();
        m.register_pass<ngraph::pass::ConstantFolding>();
        m.run_passes(f);
    }

    {
        auto data = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 1, 2}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
        auto input_low = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-128.0f});
        auto input_high = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {127.0f});
        auto output_low = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-10.0f});
        auto output_high = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1265.0f});
        size_t levels = 256;
        auto fake_q = std::make_shared<ngraph::opset1::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fake_q}, ngraph::ParameterVector{});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
