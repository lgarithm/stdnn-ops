#pragma once
#include <experimental/contract>
#include <experimental/new_type>
#include <nn/bits/engines/linag.hpp>
#include <nn/bits/ops/combinators.hpp>
#include <nn/bits/ops/im2col.hpp>
#include <nn/bits/ops/reshape.hpp>
#include <nn/bits/ops/traits.hpp>
#include <stdtensor>

namespace nn::ops
{
template <typename dim_t> class linear_conv_trait
{
    static constexpr dim_t default_rate = 1;
    static constexpr dim_t default_stride = 1;
    static constexpr dim_t default_pad_lr = 0;

    const dim_t pad_l_;
    const dim_t pad_r_;
    const dim_t rate_;
    const dim_t stride_;

  public:
    using padding_t = typename linear_sample_trait<dim_t>::padding_t;

    static padding_t padding(int p) { return padding_t(p, p); }

    static padding_t padding(int left, int right)
    {
        return padding_t(left, right);
    };

    linear_conv_trait() : linear_conv_trait(default_pad_lr) {}

    linear_conv_trait(dim_t pad_lr) : linear_conv_trait(pad_lr, default_stride)
    {
    }

    linear_conv_trait(dim_t pad_lr, dim_t stride)
        : linear_conv_trait(pad_lr, stride, default_rate)
    {
    }

    linear_conv_trait(dim_t pad_lr, dim_t stride, dim_t rate)
        : linear_conv_trait(paddint(pad_lr), stride, rate)
    {
    }

    linear_conv_trait(const padding_t &pad, dim_t stride, dim_t rate)
        : pad_l_(std::get<0>(pad.dims)),
          pad_r_(std::get<1>(pad.dims)),
          rate_(rate),
          stride_(stride)
    {
    }

    padding_t get_padding() const { return padding_t(pad_l_, pad_r_); }

    dim_t get_stride() const { return stride_; }

    dim_t get_rate() const { return rate_; }

    dim_t operator()(dim_t n, dim_t k) const
    {
        return linear_sample_trait(k, stride_, rate_, pad_l_, pad_r_)(n);
    }
};

template <typename image_order> class conv_trait;

template <> class conv_trait<hw>
{
    using dim_t = size_t;
    using conv_trait_1d_t = linear_conv_trait<dim_t>;
    using sample_t = linear_sample_trait<dim_t>;
    using padding_1d_t = sample_t::padding_t;

  protected:
    struct stride_trait;
    struct rate_trait;

    using padding_t = std::array<padding_1d_t, 2>;

    using stride_t = std::experimental::new_type<shape<2>, stride_trait>;
    using rate_t = std::experimental::new_type<shape<2>, rate_trait>;

    static constexpr auto default_stride = stride_t(1, 1);
    static constexpr auto default_rate = rate_t(1, 1);

    const conv_trait_1d_t h_trait_;
    const conv_trait_1d_t w_trait_;

    static padding_t default_padding() { return padding(0, 0); }

  public:
    static padding_1d_t padding_1d(dim_t p) { return padding_1d_t(p, p); }

    static padding_1d_t padding_1d(dim_t left, dim_t right)
    {
        return padding_1d_t(left, right);
    }

    static padding_t padding(dim_t r, dim_t s)
    {
        return padding(padding_1d(r), padding_1d(s));
    };

    static padding_t padding(const padding_1d_t &r, const padding_1d_t &s)
    {
        return {r, s};
    };

    static stride_t stride(int r, int s) { return stride_t(r, s); };

    static rate_t rate(int r, int s) { return rate_t(r, s); };

    conv_trait() : conv_trait(default_padding()) {}

    conv_trait(const padding_t &padding) : conv_trait(padding, default_stride)
    {
    }

    conv_trait(const stride_t &stride) : conv_trait(default_padding(), stride)
    {
    }

    conv_trait(const padding_t &padding, const stride_t &stride)
        : conv_trait(padding, stride, default_rate)
    {
    }

    conv_trait(const padding_t &padding, const stride_t &stride,
               const rate_t &rate)
        : h_trait_(padding[0], stride.dims[0], rate.dims[0]),
          w_trait_(padding[1], stride.dims[1], rate.dims[1])
    {
    }

    conv_trait(const conv_trait_1d_t &h_trait, const conv_trait_1d_t &w_trait)
        : h_trait_(h_trait), w_trait_(w_trait)
    {
    }

    template <typename image_order, typename filter_order>
    shape<4> infer(const shape<4> &x, const shape<4> &y) const
    {
        const auto n = batch_size<image_order>(x);
        const auto hw = image_shape<image_order>(x);
        const auto c = channel_size<image_order>(x);

        const auto rs = filter_shape<filter_order>(y);
        contract_assert(filter_in_channel_size<filter_order>(y) == c);
        const auto d = filter_out_channel_size<filter_order>(y);

        const auto h_ = h_trait_(std::get<0>(hw.dims), std::get<0>(rs.dims));
        const auto w_ = w_trait_(std::get<1>(hw.dims), std::get<1>(rs.dims));

        return batched_image_shape<image_order>(n, shape<2>(h_, w_), d);
    }
};

template <typename image_order = nhwc, typename filter_order = rscd> class conv;

template <> class conv<nhwc, rscd> : public conv_trait<hw>
{
    using conv_trait::conv_trait;

  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        return conv_trait::infer<nhwc, rscd>(x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
        // [n, h, w, c], [r, s, c, d] -> [n, h', w', d]
        // [n, h', w', r, s, c], [r, s, c, d] -> [n, h', w', d]
        // [n, h, w, c] -> [n, h', w', r, s, c]
        const auto [r, s] = filter_shape<rscd>(y.shape()).dims;

        using upper_op = im2col<hwc, hwrsc>;
        const auto upper = internal::make_batched(upper_op(
            upper_op::ksize(r, s),
            upper_op::padding(h_trait_.get_padding(), w_trait_.get_padding()),
            upper_op::stride(h_trait_.get_stride(), w_trait_.get_stride()),
            upper_op::rate(h_trait_.get_rate(), w_trait_.get_rate())));

        ttl::tensor<R, 6> x_upper(upper(x.shape()));
        upper(ref(x_upper), view(x));

        nn::engines::linag<R>::mm(
            as_matrix<3, 3, ttl::tensor_view<R, 2>>(x_upper),
            as_matrix<3, 1, ttl::tensor_view<R, 2>>(y),
            as_matrix<3, 1, ttl::tensor_ref<R, 2>>(z));
    }
};

template <> class conv<nchw, dcrs> : public conv_trait<hw>
{
    using conv_trait::conv_trait;

  public:
    shape<4> operator()(const shape<4> &x, const shape<4> &y) const
    {
        return conv_trait::infer<nchw, dcrs>(x, y);
    }

    template <typename R>
    void operator()(const ttl::tensor_ref<R, 4> &z,
                    const ttl::tensor_view<R, 4> &x,
                    const ttl::tensor_view<R, 4> &y) const
    {
        using upper_op = im2col<hw, rshw>;
        const auto [r, s] = filter_shape<dcrs>(y.shape()).dims;
        const auto upper = internal::make_batched(upper_op(
            upper_op::ksize(r, s),
            upper_op::padding(h_trait_.get_padding(), w_trait_.get_padding()),
            upper_op::stride(h_trait_.get_stride(), w_trait_.get_stride()),
            upper_op::rate(h_trait_.get_rate(), w_trait_.get_rate())));
        ttl::tensor<R, 5> x_upper(upper(x.shape().template subshape<1>()));
        const auto n = batch_size<nchw>(z.shape());
        for (auto l : range(n)) {
            upper(ref(x_upper), view(x[l]));
            nn::engines::linag<R>::mm(
                as_matrix<1, 3, ttl::tensor_view<R, 2>>(y),
                as_matrix<3, 2, ttl::tensor_view<R, 2>>(x_upper),
                as_matrix<1, 2, ttl::tensor_ref<R, 2>>(z[l]));
        }
    }
};

}  // namespace nn::ops
