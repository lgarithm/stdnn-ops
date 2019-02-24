// #define STDNN_OPS_HAVE_CBLAS

#include <algorithm>
#include <string>

#include <nn/models>

#include "utils.hpp"

using std::experimental::range;

// pool_max_3x3s2 = Pool(ksize=(3, 3), strides=(2, 2));
// pool_ave_7x7 = Pool(ksize=(7, 7), strides=(7, 7), algo=MEAN);

template <typename image_order = nn::ops::nhwc,
          typename filter_order = nn::ops::rscd>
class plain34_model
{
    const size_t logits = 1000;

    using relu = nn::ops::pointwise<nn::ops::relu>;
    using pool_ave = nn::layers::pool<nn::ops::pool_mean, image_order>;

    using dense = nn::layers::dense<>;
    using softmax = nn::layers::activation<nn::ops::softmax>;
    using top = nn::ops::top;

    auto conv1(int d) const
    {
        using conv_layer =
            nn::layers::conv<image_order, filter_order, true, relu>;
        using conv_trait = nn::ops::conv_trait<nn::ops::hw>;

        return conv_layer(
            conv_layer::ksize(7, 7), d,
            conv_trait(conv_trait::padding(conv_trait::padding_1d(3, 2),
                                           conv_trait::padding_1d(3, 2)),
                       conv_trait::stride(2, 2)));
    }

    auto pool1() const
    {
        using pool_max = nn::layers::pool<nn::ops::pool_max, image_order>;
        const auto padding = pool_max::sample1d_t::same_padding(3, 2, 1, 112);
        return pool_max(pool_max::ksize(3, 3),
                        pool_max::padding(padding, padding),
                        pool_max::stride(2, 2));
    }

    auto pool2() const
    {
        using pool_mean = nn::layers::pool<nn::ops::pool_mean, image_order>;
        return pool_mean(pool_mean::ksize(7, 7));
    }

    using conv_trait = nn::ops::conv_trait<nn::ops::hw>;

    auto conv(int d, int s, const conv_trait::padding_1d_t &padding) const
    {
        using conv_layer =
            nn::layers::conv<image_order, filter_order, true, relu>;
        return conv_layer(conv_layer::ksize(3, 3), d,
                          conv_trait(conv_trait::padding(padding, padding),
                                     conv_trait::stride(s, s)));
    }

    const std::string prefix_;

    const auto p(const std::string &name) const
    {
        return nn::ops::readtar(prefix_, name);
    }

  public:
    const size_t h = 224;
    const size_t w = 224;

    plain34_model(const std::string &prefix) : prefix_(prefix) {}

    template <typename R>
    auto operator()(const ttl::tensor_ref<R, 4> &x, int m = 5) const
    {
        auto conv_layers = nn::models::make_sequential()  //
                           << conv1(64)                   //
                           << pool1()                     //

                           << conv(64, 1, conv_trait::padding_1d(1, 1))
                           << conv(64, 1, conv_trait::padding_1d(1, 1))
                           << conv(64, 1, conv_trait::padding_1d(1, 1))
                           << conv(64, 1, conv_trait::padding_1d(1, 1))
                           << conv(64, 1, conv_trait::padding_1d(1, 1))
                           << conv(64, 1, conv_trait::padding_1d(1, 1))

                           << conv(128, 2, conv_trait::padding_1d(0, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(128, 1, conv_trait::padding_1d(1, 1))  //

                           << conv(256, 2, conv_trait::padding_1d(0, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(256, 1, conv_trait::padding_1d(1, 1))  //

                           << conv(512, 2, conv_trait::padding_1d(0, 1))  //
                           << conv(512, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(512, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(512, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(512, 1, conv_trait::padding_1d(1, 1))  //
                           << conv(512, 1, conv_trait::padding_1d(1, 1))  //

                           << pool2()  //
            ;

        auto dense_layers = nn::models::make_sequential()  //
                            << dense(logits)               //
                            << softmax()                   //
            ;

        auto y1 = conv_layers(x);
        auto y2 = nn::ops::as_matrix<1, 3, ttl::tensor_ref<R, 2>>(*y1);
        auto y = dense_layers(y2);
        return y;
    }
};

int main(int argc, char *argv[])
{
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/models/vgg16";
    plain34_model model(prefix);
    const auto x = ttl::tensor<float, 4>(1, model.h, model.w, 3);
    const auto y = model(ref(x));
    PPRINT(x);
    PPRINT(*y);
    return 0;
}