#pragma once
#include <memory>

#include <ttl/nn/common.hpp>

namespace ttl::nn::layers
{
template <typename T, typename... Ts>
class layer
{
    std::unique_ptr<T> output_;

    std::tuple<std::unique_ptr<Ts>...> weights_;

  public:
    static constexpr rank_t arity = sizeof...(Ts);

    layer(T *output, Ts *... w)
        : output_(output), weights_(std::unique_ptr<Ts>(w)...)
    {
    }

    const T &operator*() const { return *output_; }

    template <rank_t r>
    const auto &arg() const
    {
        return *std::get<r>(weights_);
    }
};

template <typename T, typename... Ts>
layer<T, Ts...> make_layer(T *y, Ts *... args)
{
    return layer<T, Ts...>(y, args...);
}

struct builder {
};
}  // namespace ttl::nn::layers
