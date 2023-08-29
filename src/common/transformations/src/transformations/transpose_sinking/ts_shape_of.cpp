// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_shape_of.hpp"

#include "itt.hpp"
#include "openvino/op/shape_of.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::pass::pattern;
using namespace ov::op::util;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using NodePair = std::pair<NodePtr, NodePtr>;

}  // namespace

TSShapeOfForward::TSShapeOfForward() {
    MATCHER_SCOPE(TSShapeOfForward);

    create_pattern<ov::op::v0::ShapeOf, ov::op::v3::ShapeOf>(true);
    auto sinking_transformation = [=](const std::shared_ptr<Node>& main_node,
                                      const TransposeInputsInfo& transpose_info) -> bool {
        main_node->input(0).replace_source_output(transpose_info.transpose->input_value(0));
        auto shape_of_consumers = main_node->output(0).get_target_inputs();
        const auto transpose_order = transpose_info.transpose_const->get_axis_vector_val();
        const auto indices = op::v0::Constant::create(element::i32, Shape{transpose_order.size()}, transpose_order);
        const auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});
        const auto gather = std::make_shared<op::v8::Gather>(main_node, indices, axis);
        for (auto& input : shape_of_consumers)
            input.replace_source_output(gather);

        return true;
    };

    transpose_sinking(matcher_name, sinking_transformation);
}
