// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/common_optimizations/conv_mul_fusion.hpp"
#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"
#include "transformations/op_conversions/batch_norm_decomposition.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvolutionBatchNorm) {
    // Model to be transformed by following transformations BatchNormDecomposition, LinOpSequenceFusion,
    // ConvolutionMultiplyFusion, ConstantFolding
    //     Convolution->BatchNorm
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto weights = op::v0::Constant::create(element::f32, Shape{4, 3, 1, 1}, {2});
        auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                              weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1});
        auto gamma = op::v0::Constant::create(element::f32, Shape{4}, {3});
        auto beta = op::v0::Constant::create(element::f32, Shape{4}, {4});
        auto mean = op::v0::Constant::create(element::f32, Shape{4}, {5});
        auto var = op::v0::Constant::create(element::f32, Shape{4}, {3});
        auto batch_norm = std::make_shared<ov::op::v5::BatchNormInference>(conv, gamma, beta, mean, var, 0.00001);
        model = std::make_shared<Model>(batch_norm, ParameterVector{data});

        manager.register_pass<ov::pass::BatchNormDecomposition>();
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
        manager.register_pass<ov::pass::ConvolutionMultiplyFusion>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    // Reference model:
    //     Convolution->Add
    {
        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto weights = op::v0::Constant::create(element::f32, Shape{4, 3, 1, 1}, {3.4641});
        auto conv = std::make_shared<ov::op::v1::Convolution>(data,
                                                              weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1});
        auto bias = op::v0::Constant::create(element::f32, Shape{1, 4, 1, 1}, {-4.66024});
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
        model_ref = std::make_shared<Model>(add, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}

TEST_F(TransformationTestsF, SubgraphConvolutionBatchNorm) {
    // Model to be transformed by following transformations BatchNormDecomposition, LinOpSequenceFusion,
    // ConvolutionMultiplyFusion, ConstantFolding
    //     Loop with Convolution->BatchNorm
    {
        /*
            let's pretend Loop below is a FunctionCall:

            FunctionCall(data_body, weights_body, gamma_body, beta_body, mean_body, var_body) {
                conv = Conv(data_body, weights_body);
                return BatchNorm(conv, gamma_body, beta_body, mean_body, var_body);
            }
        */
        auto data_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto weights_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 1, 1});
        auto conv = std::make_shared<ov::op::v1::Convolution>(data_body,
                                                              weights_body,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1});
        auto gamma_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
        auto beta_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
        auto mean_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
        auto var_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{4});
        auto batch_norm_body =
            std::make_shared<ov::op::v5::BatchNormInference>(conv, gamma_body, beta_body, mean_body, var_body, 0.00001);
        auto cond_body = op::v0::Constant::create(element::boolean, Shape{}, {true});
        auto body = std::make_shared<Model>(
            NodeVector{batch_norm_body, cond_body},
            ParameterVector{data_body, weights_body, gamma_body, beta_body, mean_body, var_body});

        auto trip_count = op::v0::Constant::create(element::i32, Shape{}, {1});
        auto cond = op::v0::Constant::create(element::boolean, Shape{}, {true});

        auto loop = std::make_shared<op::v5::Loop>(trip_count, cond);
        loop->set_function(body);
        loop->set_special_body_ports({-1, 1});

        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto weights = op::v0::Constant::create(element::f32, Shape{4, 3, 1, 1}, {2});
        auto gamma = op::v0::Constant::create(element::f32, Shape{4}, {3});
        auto beta = op::v0::Constant::create(element::f32, Shape{4}, {4});
        auto mean = op::v0::Constant::create(element::f32, Shape{4}, {5});
        auto var = op::v0::Constant::create(element::f32, Shape{4}, {3});

        /*
            "Call" a function:
            batch_norm = FunctionCall(data, weights, gamma, beta, mean, var);
        */
        loop->set_invariant_input(data_body, data);
        loop->set_invariant_input(weights_body, weights);
        loop->set_invariant_input(gamma_body, gamma);
        loop->set_invariant_input(beta_body, beta);
        loop->set_invariant_input(mean_body, mean);
        loop->set_invariant_input(var_body, var);
        auto batch_norm = loop->get_iter_value(body->get_results()[0]);

        model = std::make_shared<Model>(OutputVector{batch_norm}, ParameterVector{data});

        manager.register_pass<ov::pass::BatchNormDecomposition>();
        manager.register_pass<ov::pass::LinOpSequenceFusion>();
        manager.register_pass<ov::pass::ConvolutionMultiplyFusion>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    // Reference model:
    //     Loop with Convolution->Add
    {
        /*
            FunctionCall(data_body, weights_body, bias_body) {
                conv = Conv(data_body, weights_body);
                return Add(conv, bias_body);
            }
        */

        auto data_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto weights_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{4, 3, 1, 1});
        auto conv = std::make_shared<ov::op::v1::Convolution>(data_body,
                                                              weights_body,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1});
        auto bias_body = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 1, 1});
        auto add_body = std::make_shared<ov::op::v1::Add>(conv, bias_body);
        auto cond_body = op::v0::Constant::create(element::boolean, Shape{}, {true});
        auto body = std::make_shared<Model>(NodeVector{add_body, cond_body},
                                            ParameterVector{data_body, weights_body, bias_body});

        auto trip_count = op::v0::Constant::create(element::i32, Shape{}, {1});
        auto cond = op::v0::Constant::create(element::boolean, Shape{}, {true});

        auto loop = std::make_shared<op::v5::Loop>(trip_count, cond);
        loop->set_function(body);
        loop->set_special_body_ports({-1, 1});

        auto data = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 10, 10});
        auto weights = op::v0::Constant::create(element::f32, Shape{4, 3, 1, 1}, {3.4641});
        auto bias = op::v0::Constant::create(element::f32, Shape{1, 4, 1, 1}, {-4.66024});

        // add = FunctionCall(data, weights, bias);
        loop->set_invariant_input(data_body, data);
        loop->set_invariant_input(weights_body, weights);
        loop->set_invariant_input(bias_body, bias);
        auto add = loop->get_iter_value(body->get_results()[0]);

        model_ref = std::make_shared<Model>(OutputVector{add}, ParameterVector{data});
    }

    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
}
