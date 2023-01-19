#include "transformations/common_optimizations/push_constant_to_subgraphs.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>
#include "itt.hpp"

ov::pass::PushConstantToSubgraphs::PushConstantToSubgraphs() {
    MATCHER_SCOPE(PushConstantToSubgraphs);

    auto root = ngraph::pattern::wrap_type<op::util::MultiSubGraphOp>();
    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto subgraph = as_type_ptr<op::util::MultiSubGraphOp>(m.get_match_root());
        if (!subgraph) {
            return false;
        }

        const auto num_inputs = subgraph->get_input_size();
        OutputVector inputs;
        inputs.reserve(num_inputs);
        std::vector<std::pair<size_t, std::shared_ptr<op::v0::Constant>>> constants;
        for (size_t i = 0; i < num_inputs; i++) {
            const auto input = subgraph->input_value(i);
            auto constant = constantfold_subgraph(input);
            if (constant) {
                constant->set_friendly_name(input.get_node()->get_friendly_name());
                constants.emplace_back(i, constant);
                inputs.push_back(constant);
            } else {
                inputs.push_back(input);
            }
        }

        if (constants.size() == 0) {
            return false;
        }

        std::cout << subgraph << std::endl;

        for (size_t body_idx = 0; body_idx < subgraph->get_internal_subgraphs_size(); body_idx++) {
            const auto& body = subgraph->get_function(body_idx);
            for (auto n : body->get_ordered_ops())
                std::cout << "body " << n << std::endl;
            auto& body_params = body->get_parameters();
            auto descriptions = subgraph->get_input_descriptions(body_idx);
            for (const auto& pair : constants) {
                const auto idx = pair.first;
                const auto& constant = pair.second;
                auto it = std::find_if(descriptions.begin(), descriptions.end(),
                                       [idx] (const op::util::MultiSubGraphOp::InputDescription::Ptr& desc) {
                                           return desc->m_input_index == idx;
                                       });
                if (it == descriptions.end())
                    continue;
                const auto body_param_idx = (*it)->m_body_parameter_index;
                auto& body_param = body_params[body_param_idx];
                std::cout << "body_param " << body_param << " constant " << constant << " idx " << idx << " body_param_id " << body_param_idx << std::endl;
                body_param->output(0).replace(constant);
                body->remove_parameter(body_param);
                descriptions.erase(it);
                for (auto& desc : descriptions) {
                    if (desc->m_input_index > idx) {
                        desc->m_input_index--;
                        it++;
                    }
                    if (desc->m_body_parameter_index > body_param_idx) {
                        desc->m_body_parameter_index--;
                    }
                }
            }
            subgraph->set_input_descriptions(body_idx, descriptions);
            subgraph->set_arguments(inputs);
        }

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, matcher_name);
    register_matcher(m, callback);
}
