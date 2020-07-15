//*****************************************************************************
// Copyright 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "core/node.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/output_vector.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                inline NodeVector sin(const Node& node)
                {
                    auto input = node.get_ng_inputs().at(0);
                    auto elem_type = input->get_output_element_type(0);
                    switch (elem_type)
                    {
                    case element::Type_t::f16:
                    case element::Type_t::f32:
                    case element::Type_t::f64: break;
                    default:
                        NGRAPH_CHECK(false, "Sin operator does not support " +
                                                 elem_type.get_type_name());
                    }
                    return {std::make_shared<ngraph::op::Sin>((input))};
                }
            }; // namespace set_1
        };     // namespace op
    };         // namespace onnx_import
};             // namespace ngraph
