#pragma once
#include <experimental/contract>
#include <experimental/new_type>

#include <nn/bits/ops/linear_sample.hpp>
#include <nn/common.hpp>

namespace nn::ops
{

template <typename image_order> class pool_trait;

template <> class pool_trait<hw>
{
  protected:
    struct padding_trait;
    struct ksize_trait;
    struct stride_trait;

    using padding_t = std::experimental::new_type<shape<2>, padding_trait>;
    using ksize_t = std::experimental::new_type<shape<2>, ksize_trait>;
    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;

    static constexpr auto default_padding = padding_t(0, 0);
    static constexpr auto default_ksize = ksize_t(2, 2);

    using sample_t = linear_sample_trait<size_t>;

    const sample_t h_sample_;
    const sample_t w_sample_;

    ksize_t get_ksize() const
    {
        return ksize_t(h_sample_.ksize_, w_sample_.ksize_);
    }

    stride_t get_stride() const
    {
        return stride_t(h_sample_.stride_, w_sample_.stride_);
    }

  public:
    static padding_t padding(int r, int s) { return padding_t(r, s); };
    static ksize_t ksize(int r, int s) { return ksize_t(r, s); };
    static stride_t stride(int r, int s) { return stride_t(r, s); };

    pool_trait() : pool_trait(default_ksize) {}

    pool_trait(const ksize_t &ksize)
        : pool_trait(ksize, stride(ksize.dims[0], ksize.dims[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const stride_t &stride)
        : pool_trait(ksize, default_padding, stride)
    {
    }

    pool_trait(const ksize_t &ksize, const padding_t &padding)
        : pool_trait(ksize, padding, stride(ksize.dims[0], ksize.dims[1]))
    {
    }

    pool_trait(const ksize_t &ksize, const padding_t &padding,
               const stride_t &stride)
        : h_sample_(ksize.dims[0], stride.dims[0], 1, padding.dims[0]),
          w_sample_(ksize.dims[1], stride.dims[1], 1, padding.dims[1])
    {
    }

    shape<2> operator()(const shape<2> &x) const
    {
        return shape<2>(h_sample_(x.dims[0]), w_sample_(x.dims[1]));
    }
};

struct pool_max;
struct pool_mean;

template <typename pool_algo, typename image_order> class pool;

template <> class pool<pool_max, hw> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    //   TODO: support strided tensor
    template <typename R>
    void operator()(const ttl::tensor_ref<R, 2> &y,
                    const ttl::tensor_view<R, 2> &x) const
    {
        const auto [h, w] = x.shape().dims;
        const auto [h_, w_] = y.shape().dims;
        const auto [r, s] = get_ksize().dims;

        for (auto i_ : range(h_)) {
            for (auto j_ : range(w_)) {
                R yy = std::numeric_limits<R>::lowest();
                for (auto u : range(r)) {
                    for (auto v : range(s)) {
                        const auto i = h_sample_(i_, u);
                        const auto j = w_sample_(j_, v);
                        if (h_sample_.inside(i, h) && w_sample_.inside(j, w)) {
                            yy = std::max(yy, x.at(h_sample_.unpad(i),
                                                   w_sample_.unpad(j)));
                        }
                    }
                }
                y.at(i_, j_) = yy;
            }
        }
    }
};

template <> class pool<pool_max, hwc> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    template <typename R>
    void operator()(const ttl::tensor_ref<R, 3> &y,
                    const ttl::tensor_view<R, 3> &x) const
    {
        const auto [h, w, c] = x.shape().dims;
        const auto [h_, w_, _c] = y.shape().dims;
        const auto [r, s] = get_ksize().dims;
        contract_assert_eq(c, _c);

        for (auto k : range(c)) {
            for (auto i_ : range(h_)) {
                for (auto j_ : range(w_)) {
                    R yy = std::numeric_limits<R>::lowest();
                    for (auto u : range(r)) {
                        for (auto v : range(s)) {
                            const auto i = h_sample_(i_, u);
                            const auto j = w_sample_(j_, v);
                            if (h_sample_.inside(i, h) &&
                                w_sample_.inside(j, w)) {
                                yy = std::max(yy, x.at(h_sample_.unpad(i),
                                                       w_sample_.unpad(j), k));
                            }
                        }
                    }
                    y.at(i_, j_, k) = yy;
                }
            }
        }
    }
};

template <typename order, typename PoolTrait>
shape<4> pooled_shape(const shape<4> &x, const PoolTrait &pt)
{
    return batched_image_shape<order>(batch_size<order>(x),
                                      pt(image_shape<order>(x)),
                                      channel_size<order>(x));
}

template <> class pool<pool_max, nhwc> : public pool_trait<hw>
{
    using pool_trait::pool_trait;

  public:
    shape<4> operator()(const shape<4> &x) const
    {
        return pooled_shape<nhwc, pool_trait<hw>>(x, *this);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x) const
    {
        pool<pool_max, hwc> op(get_ksize(), get_stride());
        for (auto i : range(x.shape().dims[0])) { op(y[i], x[i]); }
    }
};

template <> class pool<pool_max, nchw> : public pool_trait<hw>
{
  public:
    using pool_trait::pool_trait;

    shape<4> operator()(const shape<4> &x) const
    {
        return pooled_shape<nchw, pool_trait<hw>>(x, *this);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &y,
                    const ttl::tensor_view<R, 4> &x) const
    {
        pool<pool_max, hw> op(get_ksize(), get_stride());
        for (auto i : range(x.shape().dims[0])) {
            for (auto j : range(x.shape().dims[1])) { op(y[i][j], x[i][j]); }
        }
    }
};

}  // namespace nn::ops
