// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ov {
namespace pass {

/**
 * @ingroup ie_transformation_common_api
 * @brief PushConstantToSubgraph transformation detects Constant nodes
 * that are inputs to MultiSubGraphOp and pushes that Constant to subgraphs.
 */
class TRANSFORMATIONS_API PushConstantToSubgraphs : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PushConstantToSubgraphs", "0");
    PushConstantToSubgraphs();
};

}  // namespace pass
}  // namespace ov
