// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/mul_conv_fusion.hpp"
#include "common_test_utils/test_constants.hpp"

#include <ngraph/opsets/opset8.hpp>

using namespace SubgraphTestsDefinitions;

namespace {
    const std::vector<ngraph::element::Type> types{ngraph::element::f32, ngraph::element::f16};

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_1D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 8, 64}),
                                    ::testing::Values(ngraph::Shape{64, 8, 1}),
                                    ::testing::Values(ngraph::Shape{8, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_2d{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {3, 1, 1},
        {1, 1, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset8::Convolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{20, 3, 1, 1}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::ConvolutionBackpropData::type_info),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{3, 20, 3, 3}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_group_conv{
        {},
        {1},
        {1, 1},
        {1, 1, 1},
        {3, 1, 1},
        {1, 1, 1, 1},
        {1, 1, 1, 1, 1},
    };

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 2}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_2D, MulConvFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolutionBackpropData::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1, 1}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            MulConvFusion::getTestCaseName);

}  // namespace
