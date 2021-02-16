#include <ttl/nn/bits/engines/config.hpp>
#include <ttl/tensor>

#include <Eigen/Core>

namespace ttl::nn::engines
{
template <typename R>
struct eigen_matrix;

template <>
struct eigen_matrix<float> {
    using type = Eigen::MatrixXf;
};

template <>
struct eigen_matrix<double> {
    using type = Eigen::MatrixXd;
};

template <typename R>
class eigen_impl
{
    using m_ref_t = ttl::matrix_ref<R>;
    using m_view_t = ttl::matrix_view<R>;
    using v_ref_t = ttl::vector_ref<R>;
    using v_view_t = ttl::vector_view<R>;

    using eigen_mat = typename eigen_matrix<R>::type;

    static eigen_mat to(const m_view_t &a)
    {
        eigen_mat m(std::get<0>(a.dims()), std::get<1>(a.dims()));
        return m;
    }

  public:
    static void mm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        // FIXME: copy elison
        const eigen_mat A = to(a);
        const eigen_mat B = to(b);
        eigen_mat C = A * B;
        std::copy(C.data(), C.data() + c.size(), c.data());
    }

    static void mmt(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        throw std::runtime_error("TODO");
    }

    static void mtm(const m_view_t &a, const m_view_t &b, const m_ref_t &c)
    {
        throw std::runtime_error("TODO");
    }

    static void mv(const m_view_t &a, const v_view_t &b, const v_ref_t &c)
    {
        throw std::runtime_error("TODO");
    }

    static void vm(const v_view_t &a, const m_view_t &b, const v_ref_t &c)
    {
        throw std::runtime_error("TODO");
    }
};

struct eigen;

template <>
struct backend<eigen> {
    template <typename R>
    using type = eigen_impl<R>;
};
}  // namespace ttl::nn::engines
