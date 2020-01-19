#pragma once
#include <cmath>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>

#include <ttl/algorithm>
#include <ttl/nn/bits/kernels/utility.hpp>

namespace ttl::nn::kernels
{
template <typename R, typename R1>
class cast<host_memory, R, R1>
{
  public:
    void operator()(const tensor_ref<R, 1> &y,
                    const tensor_view<R1, 1> &x) const
    {
        ttl::cast(x, y);
    }
};

template <typename N, typename R>
class argmax<host_memory, N, R>
{
  public:
    void operator()(const tensor_ref<N, 1> &y, const tensor_view<R, 2> &x) const
    {
        for (auto i : range(y.shape().size())) {
            y.data()[i] = ttl::argmax(x[i]);
        }
    }
};

template <typename N, typename R>
class onehot<host_memory, N, R>
{
  public:
    void operator()(const tensor_ref<R, 2> &y, const tensor_view<N, 1> &x) const
    {
        const auto k = std::get<1>(y.shape().dims());
        ttl::fill(y, static_cast<R>(0));
        for (auto i : range(x.shape().size())) {
            if (const N j = x.data()[i]; 0 <= j && j < k) {
                y.at(i, j) = static_cast<R>(1);
            } else {
                // TODO: throw?
            }
        }
    }
};

template <typename N, typename R>
class top<host_memory, N, R>
{
  public:
    void operator()(const tensor_ref<R, 1> &y, const tensor_ref<N, 1> &z,
                    const tensor_view<R, 1> &x) const
    {
        const N n = x.shape().size();
        const N k = y.shape().size();
        using P = std::pair<R, N>;
        std::vector<P> v(n);
        for (auto i : range(n)) {
            v[i].first = x.at(i);
            v[i].second = i;
        }
        std::sort(v.begin(), v.end(), std::greater<P>());
        for (auto i : range(std::min(n, k))) {
            y.at(i) = v[i].first;
            z.at(i) = v[i].second;
        }
    }
};

template <typename R, typename R1>
class similarity<host_memory, R, R1>
{
  public:
    void operator()(const tensor_ref<R, 0> &z, const tensor_view<R1, 1> &x,
                    const tensor_view<R1, 1> &y) const
    {
        static_assert(std::is_floating_point<R>::value);
        z.data()[0] = 1 - static_cast<R>(ttl::hamming_distance(x, y)) /
                              static_cast<R>(x.shape().size());
    }
};
}  // namespace ttl::nn::kernels