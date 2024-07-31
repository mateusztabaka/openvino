// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_fake_quantize.hpp"

#include "itt.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"


using namespace std;
using namespace ov;
using namespace ov::pass;


pass::RemoveFakeQuantize::RemoveFakeQuantize(const element::TypeVector& supported_types) {
    MATCHER_SCOPE(RemoveFakeQuantize);

    auto input_pattern = pattern::any_input();
    auto input_low_pattern = pattern::wrap_type<op::v0::Constant>();
    auto input_high_pattern = pattern::wrap_type<op::v0::Constant>();
    auto output_low_pattern = pattern::wrap_type<op::v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<op::v0::Constant>();
    auto fq_pattern = pattern::wrap_type<op::v0::FakeQuantize>({input_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = ov::as_type_ptr<op::v0::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        size_t levels = fq->get_levels();
        for (const auto& type : supported_types) {
            size_t bitwidth = type.bitwidth();
            if (bitwidth >= 64)
                return false;
            if (!type.is_integral())
                return false;
            size_t max_value = 1 << bitwidth;
            if (levels == max_value || levels == max_value - 1) {
                return false;
            }
        }
        auto input_low = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(1));
        auto input_high = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(2));
        auto output_low = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(3));
        auto output_high = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(4));

        auto input_low_values = input_low->cast_vector<float>();
        auto input_high_values = input_high->cast_vector<float>();
        auto output_low_values = output_low->cast_vector<float>();
        auto output_high_values = output_high->cast_vector<float>();

        if (input_low_values != output_low_values ||
            input_high_values != output_high_values) {
            return false;
        }

        return replace_output_update_name(fq->output(0), fq->input_value(0));
    };

    auto m = make_shared<pattern::Matcher>(fq_pattern, matcher_name);
    this->register_matcher(m, callback);
}
