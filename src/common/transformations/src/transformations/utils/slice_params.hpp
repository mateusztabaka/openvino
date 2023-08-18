#include <vector>
#include "openvino/core/node.hpp"
#include "ov_optional.hpp"


namespace ov {
namespace op {
namespace util {

struct SliceParams {
    std::vector<int64_t> start;
    std::vector<int64_t> stop;
    std::vector<int64_t> stride;
    std::vector<int64_t> axes;

    bool operator==(const SliceParams& other) const {
        return start == other.start &&
               stop == other.stop &&
               stride == other.stride &&
               axes == other.axes;
    }
};

ov::optional<SliceParams> get_slice_params(const Node* node);

}  // namespace util
}  // namespace op
}  // namespace ov
