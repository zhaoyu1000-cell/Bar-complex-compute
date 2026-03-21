#ifndef SPARSE_RANK_WIEDEMANN_PARALLEL_IMPL
#define SPARSE_RANK_WIEDEMANN_PARALLEL_IMPL

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "sparse_rank_wiedemann.cpp"

namespace sparse_wiedemann_parallel {

using Int = sparse_wiedemann::Int;
using SparsePairMatrix = sparse_wiedemann::SparsePairMatrix;

static int clamp_worker_count(int requested_threads) {
#ifdef _OPENMP
    int hw_threads = omp_get_max_threads();
#else
    int hw_threads = 1;
#endif
    if (hw_threads <= 0) hw_threads = 1;
    if (requested_threads <= 0) return hw_threads;
    return std::max(1, requested_threads);
}

static std::vector<Int> apply_sparse_matrix_parallel(const SparsePairMatrix& a,
                                                     const std::vector<Int>& x,
                                                     Int p,
                                                     int threads) {
    const int n = static_cast<int>(a.size());
    std::vector<Int> y(n, 0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#else
    (void)threads;
#endif
    for (int i = 0; i < n; ++i) {
        Int acc = 0;
        for (const auto& [j, val] : a[i]) {
            acc = (acc + val * x[j]) % p;
        }
        y[i] = acc;
    }

    return y;
}

static SparsePairMatrix transpose_sparse_matrix_parallel(const SparsePairMatrix& a) {
    const int n = static_cast<int>(a.size());
    SparsePairMatrix at(n);
    std::vector<int> counts(n, 0);
    for (int i = 0; i < n; ++i) {
        for (const auto& [j, _] : a[i]) ++counts[j];
    }
    for (int j = 0; j < n; ++j) at[j].reserve(counts[j]);
    for (int i = 0; i < n; ++i) {
        for (const auto& [j, v] : a[i]) at[j].push_back({i, v});
    }
    return at;
}

static Int dot_mod_parallel(const std::vector<Int>& u,
                            const std::vector<Int>& w,
                            Int p,
                            int threads) {
    const int n = static_cast<int>(u.size());
#ifdef _OPENMP
    std::vector<Int> partial(std::max(1, threads), 0);
#pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        Int local = 0;
#pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            local = (local + u[i] * w[i]) % p;
        }
        partial[tid] = local;
    }

    Int sum = 0;
    for (Int v : partial) sum = (sum + v) % p;
    return sum;
#else
    (void)threads;
    Int sum = 0;
    for (int i = 0; i < n; ++i) sum = (sum + u[i] * w[i]) % p;
    return sum;
#endif
}

static std::vector<Int> apply_preconditioned_gram_parallel(const SparsePairMatrix& a,
                                                            const SparsePairMatrix& at,
                                                            const std::vector<Int>& x,
                                                            const std::vector<Int>& d1,
                                                            const std::vector<Int>& d2,
                                                            Int p,
                                                            int threads) {
    const int n = static_cast<int>(a.size());
    std::vector<Int> tmp1(n, 0);
    std::vector<Int> tmp2(n, 0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        tmp1[i] = (d2[i] * x[i]) % p;
    }

    tmp2 = apply_sparse_matrix_parallel(a, tmp1, p, threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        tmp2[i] = (d1[i] * tmp2[i]) % p;
    }

    tmp1 = apply_sparse_matrix_parallel(at, tmp2, p, threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        tmp1[i] = (d2[i] * tmp1[i]) % p;
    }

    return tmp1;
}

// Parallelized Wiedemann rank estimator.
//
// This version parallelizes the expensive black-box operator applications and
// inner products inside each Monte Carlo trial. With OpenMP enabled
// (-fopenmp), this delivers near-linear scaling for sufficiently large n and d.
template <typename URBG>
static int rank_probabilistic_parallel_with_rng(const SparsePairMatrix& a,
                                                Int p,
                                                URBG& rng,
                                                int repeats = 3,
                                                int requested_threads = 0) {
    if (p <= 2) {
        throw std::invalid_argument("Modulus p must be an odd prime > 2");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("repeats must be positive");
    }

    const int n = static_cast<int>(a.size());
    if (n == 0) return 0;
    const SparsePairMatrix at = transpose_sparse_matrix_parallel(a);

    const int threads = clamp_worker_count(requested_threads);

    std::uniform_int_distribution<Int> nz_dist(1, p - 1);
    std::uniform_int_distribution<Int> any_dist(0, p - 1);

    int best_rank = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        std::vector<Int> d1(n), d2(n), u(n);
        for (int i = 0; i < n; ++i) {
            d1[i] = nz_dist(rng);
            d2[i] = nz_dist(rng);
            u[i] = any_dist(rng);
        }

        std::vector<Int> sequence(2 * n, 0);
        std::vector<Int> w = u;

        for (int k = 0; k < 2 * n; ++k) {
            sequence[k] = dot_mod_parallel(u, w, p, threads);
            w = apply_preconditioned_gram_parallel(a, at, w, d1, d2, p, threads);
        }

        int degree = sparse_wiedemann::berlekamp_massey_linear_complexity(sequence, p);
        int estimate = (degree == n) ? n : std::max(0, degree - 1);
        best_rank = std::max(best_rank, estimate);
        if (best_rank == n) break;
    }

    return best_rank;
}

int rank_probabilistic_parallel(const SparsePairMatrix& a,
                                Int p,
                                int repeats,
                                int requested_threads,
                                std::uint64_t seed = std::random_device{}()) {
    std::mt19937_64 rng(seed);
    return rank_probabilistic_parallel_with_rng(a, p, rng, repeats, requested_threads);
}

}  // namespace sparse_wiedemann_parallel

#endif  // SPARSE_RANK_WIEDEMANN_PARALLEL_IMPL
