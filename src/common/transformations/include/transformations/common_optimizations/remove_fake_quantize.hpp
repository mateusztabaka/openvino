// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API RemoveFakeQuantize;

}  // namespace pass
}  // namespace ov

class ov::pass::RemoveFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RemoveFakeQuantize", "0");
    RemoveFakeQuantize(const element::TypeVector& supported_types);
};
