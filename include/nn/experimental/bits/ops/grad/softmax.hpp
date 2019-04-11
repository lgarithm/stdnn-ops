#pragma once
#include <nn/bits/engines/linag.hpp>
#include <nn/common.hpp>

#include <experimental/range>

using std::experimental::range;

namespace nn::experimental::ops::grad
{
template <int> class softmax;

template <> class softmax<0>
{
  public:
    template <typename R>
    void operator()(const ttl::tensor_ref<R, 1> &gx,
                    const ttl::tensor_view<R, 1> &gy,
                    const ttl::tensor_view<R, 1> &y,
                    const ttl::tensor_view<R, 1> &x)
    {
        const auto n = x.shape().size();
        const ttl::tensor<R, 2> g(n, n);
        for (auto i : range(n)) {
            for (auto j : range(n)) {
                if (i == j) {
                    const R v = y.at(i);
                    g.at(i, j) = v * (static_cast<R>(1) - v);
                } else {
                    g.at(i, j) = -y.at(i) * y.at(j);
                }
            }
        }
        nn::engines::linag<nn::engines::default_engine>::vm(gy, view(g), gx);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &gx,
                    const ttl::tensor_view<R, 2> &gy,
                    const ttl::tensor_view<R, 2> &y,
                    const ttl::tensor_view<R, 2> &x)
    {
        for (auto i : range(x.shape().dims[0])) {
            operator()(gx[i], gy[i], y[i], x[i]);
        }
    }
};
}  // namespace nn::experimental::ops::grad
