// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"

namespace ov {
namespace pass {
namespace pattern {

template <class... NodeTypes>
std::shared_ptr<Node> optional(const OutputVector& inputs, const Output<Node>& parent, const pattern::op::ValuePredicate& pred)  {
    auto label = pattern::wrap_type<NodeTypes...>(inputs, pred);
    return std::make_shared<pattern::op::Or>(OutputVector{label, parent}, pred);
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const OutputVector& inputs, const Output<Node>& parent) {
    auto label = pattern::wrap_type<NodeTypes...>(inputs);
    return std::make_shared<pattern::op::Or>(OutputVector{label, parent});
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const Output<Node>& parent, const pattern::op::ValuePredicate& pred)  {
    return optional<NodeTypes...>({}, parent, pred);
}

template <class... NodeTypes>
std::shared_ptr<Node> optional(const Output<Node>& parent) {
    return optional<NodeTypes...>({}, parent);
}

}  // namespace pattern
}  // namespace pass
}  // namespace ov
