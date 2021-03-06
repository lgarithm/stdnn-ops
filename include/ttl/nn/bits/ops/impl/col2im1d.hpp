#pragma once
#include <ttl/algorithm>
#include <ttl/nn/bits/ops/col2im1d.hpp>

namespace ttl::nn::ops
{
template <typename R>
void col2im1d::operator()(const ttl::tensor_ref<R, 1> &y,
                          const ttl::tensor_view<R, 2> &x) const
{
    const auto [n] = y.shape().dims();
    ttl::fill(y, static_cast<R>(0));
    for (auto j : ttl::range<0>(x)) {
        for (auto k : ttl::range<1>(x)) {
            const int i = (*this)(j, k);
            if (inside(i, n)) { y.at(unpad(i)) += x.at(j, k); }
        }
    }
}
}  // namespace ttl::nn::ops
