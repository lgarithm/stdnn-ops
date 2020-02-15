#include <ttl/nn/bits/ops/col2im.hpp>
#include <ttl/nn/bits/ops/im2col.hpp>
#include <ttl/nn/bits/ops/init.hpp>

#include "benchmark.hpp"
#include "common.hpp"

template <int h, int w, int c>
struct bench_image_to_column {
    using F = ttl::nn::ops::im2col<ttl::nn::traits::hwc,  //
                                   ttl::nn::traits::hwrsc>;
    using G = ttl::nn::ops::col2im<ttl::nn::traits::hwc,  //
                                   ttl::nn::traits::hwrsc>;

    static void run_im2col(benchmark::State &state)
    {
        F f(F::ksize(3, 3), F::padding(1, 1));
        using B = bench<F, float, ttl::shape<5>, ttl::shape<3>>;
        B b(f, ttl::make_shape(h, w, c));
        b.init<0>(ttl::nn::ops::ones());
        for (auto _ : state) { b(); }
    }

    static void run_col2im(benchmark::State &state)
    {
        F f(F::ksize(3, 3), F::padding(1, 1));
        G g(G::ksize(3, 3), G::padding(1, 1));
        using B = bench<G, float, ttl::shape<3>, ttl::shape<5>>;
        B b(g, f(ttl::make_shape(h, w, c)));
        b.init<0>(ttl::nn::ops::ones());
        for (auto _ : state) { b(); }
    }
};

static void bench_im2col_hwc_hwrsc_256_384_32(benchmark::State &state)
{
    bench_image_to_column<256, 384, 32>::run_im2col(state);
}
BENCHMARK(bench_im2col_hwc_hwrsc_256_384_32)->Unit(benchmark::kMillisecond);

static void bench_col2im_hwc_hwrsc_256_384_32(benchmark::State &state)
{
    bench_image_to_column<256, 384, 32>::run_col2im(state);
}
BENCHMARK(bench_col2im_hwc_hwrsc_256_384_32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
