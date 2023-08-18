#include "transformations/utils/slice_params.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ov {
namespace op {
namespace util {

static ov::optional<std::vector<int64_t>> get_input_as_vector(const Node* node) {
    if (auto constant = ov::as_type<const v0::Constant>(node)) {
        return constant->cast_vector<int64_t>();
    }
    return {};
}

static std::vector<int64_t> get_default_axes(int64_t rank) {
    std::vector<int64_t> ret(rank);
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
}

ov::optional<SliceParams> get_slice_params(const Node* node) {
    auto strided_slice = ov::as_type<const v1::StridedSlice>(node);
    if (!ov::is_type<v8::Slice>(node) && strided_slice == nullptr)
        return {};
    const auto& input_shape = node->get_input_partial_shape(0);
    if (input_shape.rank().is_dynamic())
        return {};
    const auto rank = static_cast<int64_t>(input_shape.size());

    auto start_opt = get_input_as_vector(node->get_input_node_ptr(1));
    if (!start_opt)
        return {};

    auto stop_opt = get_input_as_vector(node->get_input_node_ptr(2));
    if (!stop_opt)
        return {};

    ov::optional<std::vector<int64_t>> stride_opt = node->get_input_size() < 4 ?
                                                    std::vector<int64_t>(rank, 1) : get_input_as_vector(node->get_input_node_ptr(3));
    if (!stride_opt)
        return {};

    ov::optional<std::vector<int64_t>> axes_opt = (strided_slice || node->get_input_size() < 5) ?
                                                get_default_axes(rank) : get_input_as_vector(node->get_input_node_ptr(4));
    if (!axes_opt)
        return {};

    const std::vector<int64_t>& begin_mask = strided_slice ? strided_slice->get_begin_mask() : std::vector<int64_t>{};
    const std::vector<int64_t>& end_mask = strided_slice ? strided_slice->get_end_mask() : std::vector<int64_t>{};
    const std::vector<int64_t>& new_axis_mask = strided_slice ? strided_slice->get_new_axis_mask() : std::vector<int64_t>{};
    if (std::find(new_axis_mask.begin(), new_axis_mask.end(), 1) != new_axis_mask.end())
        return {};
    const std::vector<int64_t>& shrink_axis_mask = strided_slice ? strided_slice->get_shrink_axis_mask() : std::vector<int64_t>{};
    if (std::find(shrink_axis_mask.begin(), shrink_axis_mask.end(), 1) != shrink_axis_mask.end())
        return {};
    const std::vector<int64_t>& ellipsis_mask = strided_slice ? strided_slice->get_ellipsis_mask() : std::vector<int64_t>{};
    if (std::find(ellipsis_mask.begin(), ellipsis_mask.end(), 1) != ellipsis_mask.end())
        return {};

    auto& start = *start_opt;
    auto& stop = *stop_opt;
    auto& stride = *stride_opt;
    auto& axes = *axes_opt;

    for (size_t i = 0; i < axes.size(); i++) {
        auto& axis = axes[i];
        if (axis < 0) {
                axis += rank;
        }

        const auto& dim = input_shape[axis];
        bool dim_is_static = dim.is_static();

        if (i < begin_mask.size() && begin_mask[i] == 1) {
            start[i] = 0;
        } else {
            if (start[i] < 0 && dim_is_static)
                start[i] += dim.get_length();
        }

        if (i < end_mask.size() && end_mask[i] == 1) {
            stop[i] = -1;
        } else {
            if (stop[i] < 0 && dim_is_static)
                stop[i] += dim.get_length();
        }

        if (dim_is_static) {
            start[i] = std::min(start[i], dim.get_length());
            stop[i] = std::min(stop[i], dim.get_length());
        }

        if (start[i] == 0 && (stop[i] == -1 || (dim_is_static && stop[i] == dim.get_length())) && stride[i] == 1) {
            axes.erase(axes.begin() + i);
            start.erase(start.begin() + i);
            stop.erase(stop.begin() + i);
            stride.erase(stride.begin() + i);
        } else {
            i++;
        }
    }

    return SliceParams{ start, stop, stride, axes };
}

}  // namespace util
}  // namespace op
}  // namespace ov
