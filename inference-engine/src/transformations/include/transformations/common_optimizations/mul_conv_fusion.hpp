// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <functional>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset8.hpp>

namespace ngraph {
namespace pass {

template <typename T>
class TRANSFORMATIONS_API MultiplyConvolutionFusion;

template <>
class TRANSFORMATIONS_API MultiplyConvolutionFusion<opset8::Convolution>;

}  // namespace pass
}  // namespace ngraph

template <typename T>
class ngraph::pass::MultiplyConvolutionFusion: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    MultiplyConvolutionFusion();
};
