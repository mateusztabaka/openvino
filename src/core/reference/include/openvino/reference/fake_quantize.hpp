// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include "ngraph/check.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/shape_util.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
namespace fake_quantize_details {
template <typename T>
inline T quantize(const T arg,
                  const T in_low,
                  const T in_high,
                  const T out_low,
                  const T out_high,
                  const T levels_minus_one) {
    if (arg <= std::min(in_low, in_high)) {
        return out_low;
    } else if (arg > std::max(in_low, in_high)) {
        return out_high;
    }
    return static_cast<T>(std::nearbyint((arg - in_low) / (in_high - in_low) * levels_minus_one) / levels_minus_one *
                              (out_high - out_low) +
                          out_low);
}

template <typename T, typename F>
static void transform(const T* first1, const T* const last1, const T* first2, const T* first3, T* out, const F& f) {
    while (first1 < last1) {
        *out++ = f(*first1++, *first2++, *first3++);
    }
}

template <typename T, typename F>
static void transform(const T* first1,
                      const T* const last1,
                      const T* first2,
                      const T* first3,
                      const T* first4,
                      const T* first5,
                      T* out,
                      const F& f) {
    while (first1 < last1) {
        *out++ = f(*first1++, *first2++, *first3++, *first4++, *first5++);
    }
}

template <typename T, typename F1, typename F2>
static void fake_quantize_loop(const Shape& arg_shape,
                               const T* arg,
                               const T* in_low,
                               const T* in_high,
                               const T* out_low,
                               const T* out_high,
                               T* out,
                               size_t input_inner_stride,
                               const F1& get_outer_strides,
                               const F2& quantize_loop) {
    size_t in_low_stride = 0;
    size_t in_high_stride = 0;
    size_t out_low_stride = 0;
    size_t out_high_stride = 0;

    for (size_t i = 0; i < shape_size(arg_shape); i += input_inner_stride) {
        std::tie(in_low_stride, in_high_stride, out_low_stride, out_high_stride) = get_outer_strides(i);
        quantize_loop(arg,
                      arg + input_inner_stride,
                      in_low + in_low_stride,
                      in_high + in_high_stride,
                      out_low + out_low_stride,
                      out_high + out_high_stride,
                      out);
        arg += input_inner_stride;
        out += input_inner_stride;
    }
}

}  // namespace fake_quantize_details

template <typename T>
void fake_quantize(const T* arg,
                   const T* in_low,
                   const T* in_high,
                   const T* out_low,
                   const T* out_high,
                   T* out,
                   const Shape& arg_shape,
                   const Shape& in_low_shape,
                   const Shape& in_high_shape,
                   const Shape& out_low_shape,
                   const Shape& out_high_shape,
                   size_t levels,
                   const op::AutoBroadcastSpec& broadcast) {
    using namespace fake_quantize_details;

    T levels_minus_one = static_cast<T>(levels - 1);

    if (shape_size(in_low_shape) == 1 && shape_size(in_high_shape) == 1 && shape_size(out_low_shape) == 1 &&
        shape_size(out_high_shape) == 1) {
        const size_t arg_size = shape_size(arg_shape);
        const auto q = [&](const T& a) {
            return quantize(a, *in_low, *in_high, *out_low, *out_high, levels_minus_one);
        };
        for (size_t i = 0; i < arg_size; ++i) {
            out[i] = q(arg[i]);
        }
        return;
    }

    auto compute_strides = [](const Shape& out_shape, const Shape& shape) {
        size_t stride = 1;
        size_t out_rank = out_shape.size();
        size_t shape_rank = shape.size();
        std::vector<size_t> strides(out_rank);
        for (size_t i = 0; i < out_rank; i++) {
            if (i < shape_rank && shape[shape_rank - i - 1] == out_shape[out_rank - i - 1]) {
                strides[out_rank - i - 1] = stride;
                stride *= shape[shape_rank - i - 1];
            } else {
                strides[out_rank - i - 1] = 0;
            }
        }
        return strides;
    };

    std::vector<size_t> output_strides = compute_strides(arg_shape, arg_shape);
    std::vector<size_t> in_low_strides = compute_strides(arg_shape, in_low_shape);
    std::vector<size_t> in_high_strides = compute_strides(arg_shape, in_high_shape);
    std::vector<size_t> out_low_strides = compute_strides(arg_shape, out_low_shape);
    std::vector<size_t> out_high_strides = compute_strides(arg_shape, out_high_shape);

    size_t num_elements = shape_size(arg_shape);
    auto get_inner_stride = [num_elements, &arg_shape](const Shape& interval_shape, size_t current_input_inner_stride) {
        if (interval_shape.size() == 0)
            return std::tuple<size_t, size_t>{1, std::min(current_input_inner_stride, num_elements)};
        size_t last = interval_shape.back();
        auto it = std::find_if(interval_shape.rbegin(), interval_shape.rend(), [last](size_t dim) {
            return (last == 1 && dim > 1) || (last > 1 && dim == 1);
        });
        if (it == interval_shape.rend())
            return std::tuple<size_t, size_t>{last == 1 ? 1 : num_elements,
                                              std::min(current_input_inner_stride, num_elements)};
        size_t idx = std::distance(it, interval_shape.rbegin()) + static_cast<int64_t>(interval_shape.size());
        size_t interval_inner_stride =
            std::accumulate(interval_shape.begin() + idx, interval_shape.end(), 1, std::multiplies<size_t>());
        size_t input_inner_stride = std::accumulate(arg_shape.begin() + arg_shape.size() - interval_shape.size() + idx,
                                                    arg_shape.end(),
                                                    1,
                                                    std::multiplies<size_t>());
        return std::tuple<size_t, size_t>{interval_inner_stride,
                                          std::min(current_input_inner_stride, input_inner_stride)};
    };

    size_t input_inner_stride = num_elements;
    size_t in_low_inner_stride = 0;
    size_t in_high_inner_stride = 0;
    size_t out_low_inner_stride = 0;
    size_t out_high_inner_stride = 0;

    std::tie(in_low_inner_stride, input_inner_stride) = get_inner_stride(in_low_shape, input_inner_stride);
    std::tie(in_high_inner_stride, input_inner_stride) = get_inner_stride(in_high_shape, input_inner_stride);
    std::tie(out_low_inner_stride, input_inner_stride) = get_inner_stride(out_low_shape, input_inner_stride);
    std::tie(out_high_inner_stride, input_inner_stride) = get_inner_stride(out_high_shape, input_inner_stride);

    auto get_outer_strides =
        [&output_strides, &in_low_strides, &in_high_strides, &out_low_strides, &out_high_strides](size_t flat_index) {
            size_t in_low_stride = 0;
            size_t in_high_stride = 0;
            size_t out_low_stride = 0;
            size_t out_high_stride = 0;

            for (size_t i = 0; i < output_strides.size(); i++) {
                size_t div = flat_index / output_strides[i];
                flat_index = flat_index % output_strides[i];
                in_low_stride += div * in_low_strides[i];
                in_high_stride += div * in_high_strides[i];
                out_low_stride += div * out_low_strides[i];
                out_high_stride += div * out_high_strides[i];
            }

            return std::tuple<size_t, size_t, size_t, size_t>{in_low_stride,
                                                              in_high_stride,
                                                              out_low_stride,
                                                              out_high_stride};
        };

    size_t in_low_stride = 0;
    size_t in_high_stride = 0;
    size_t out_low_stride = 0;
    size_t out_high_stride = 0;

    enum class IntervalsType {
        NON_SCALAR = 0,
        SCALAR = 0b1111,
        INPUT_SCALAR = 0b1100,
        OUTPUT_SCALAR = 0b0011,
    };

    IntervalsType intervals_type =
        static_cast<IntervalsType>(((in_low_inner_stride == 1) << 3) | ((in_high_inner_stride == 1) << 2) |
                                   ((out_low_inner_stride == 1) << 1) | (out_high_inner_stride == 1));

    switch (intervals_type) {
    case IntervalsType::NON_SCALAR: {
        fake_quantize_loop(arg_shape,
                           arg,
                           in_low,
                           in_high,
                           out_low,
                           out_high,
                           out,
                           input_inner_stride,
                           get_outer_strides,
                           [levels_minus_one](const T* input,
                                              const T* const input_end,
                                              const T* in_low,
                                              const T* in_high,
                                              const T* out_low,
                                              const T* out_high,
                                              T* out) {
                               transform(
                                   input,
                                   input_end,
                                   in_low,
                                   in_high,
                                   out_low,
                                   out_high,
                                   out,
                                   [levels_minus_one](T input, T in_low, T in_high, T out_low, T out_high) {
                                       return quantize(input, in_low, in_high, out_low, out_high, levels_minus_one);
                                   });
                           });
        break;
    }
    case IntervalsType::SCALAR: {
        auto quantize_with_scalar_intervals = [levels_minus_one](const T* input,
                                                                 const T* const input_end,
                                                                 const T* in_low,
                                                                 const T* in_high,
                                                                 const T* out_low,
                                                                 const T* out_high,
                                                                 T* out) {
            auto in_low_scalar = *in_low;
            auto in_high_scalar = *in_high;
            auto out_low_scalar = *out_low;
            auto out_high_scalar = *out_high;
            std::transform(input,
                           input_end,
                           out,
                           [levels_minus_one, in_low_scalar, in_high_scalar, out_low_scalar, out_high_scalar](T input) {
                               return quantize(input,
                                               in_low_scalar,
                                               in_high_scalar,
                                               out_low_scalar,
                                               out_high_scalar,
                                               levels_minus_one);
                           });
        };

        fake_quantize_loop(arg_shape,
                           arg,
                           in_low,
                           in_high,
                           out_low,
                           out_high,
                           out,
                           input_inner_stride,
                           get_outer_strides,
                           quantize_with_scalar_intervals);
        break;
    }
    case IntervalsType::INPUT_SCALAR: {
        auto quantize_with_scalar_input_intervals = [levels_minus_one](const T* input,
                                                                       const T* const input_end,
                                                                       const T* in_low,
                                                                       const T* in_high,
                                                                       const T* out_low,
                                                                       const T* out_high,
                                                                       T* out) {
            auto in_low_scalar = *in_low;
            auto in_high_scalar = *in_high;
            transform(input,
                      input_end,
                      out_low,
                      out_high,
                      out,
                      [levels_minus_one, in_low_scalar, in_high_scalar](T input, T out_low, T out_high) {
                          return quantize(input, in_low_scalar, in_high_scalar, out_low, out_high, levels_minus_one);
                      });
        };

        fake_quantize_loop(arg_shape,
                           arg,
                           in_low,
                           in_high,
                           out_low,
                           out_high,
                           out,
                           input_inner_stride,
                           get_outer_strides,
                           quantize_with_scalar_input_intervals);

        break;
    }
    case IntervalsType::OUTPUT_SCALAR: {
        auto quantize_with_scalar_output_intervals = [levels_minus_one](const T* input,
                                                                        const T* const input_end,
                                                                        const T* in_low,
                                                                        const T* in_high,
                                                                        const T* out_low,
                                                                        const T* out_high,
                                                                        T* out) {
            auto out_low_scalar = *out_low;
            auto out_high_scalar = *out_high;
            transform(input,
                      input_end,
                      out_low,
                      out_high,
                      out,
                      [levels_minus_one, out_low_scalar, out_high_scalar](T input, T in_low, T in_high) {
                          return quantize(input, in_low, in_high, out_low_scalar, out_high_scalar, levels_minus_one);
                      });
        };

        fake_quantize_loop(arg_shape,
                           arg,
                           in_low,
                           in_high,
                           out_low,
                           out_high,
                           out,
                           input_inner_stride,
                           get_outer_strides,
                           quantize_with_scalar_output_intervals);

        break;
    }
    default: {
        for (size_t i = 0; i < shape_size(arg_shape); i++) {
            std::tie(in_low_stride, in_high_stride, out_low_stride, out_high_stride) = get_outer_strides(i);
            *out++ = quantize(*arg++,
                              *(in_low + in_low_stride),
                              *(in_high + in_high_stride),
                              *(out_low + out_low_stride),
                              *(out_high + out_low_stride),
                              levels_minus_one);
        }
        break;
    }
    }
}

}  // namespace reference
}  // namespace ov
