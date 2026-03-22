#ifndef SPARSE_RANK_WIEDEMANN_PARALLEL_IMPL
#define SPARSE_RANK_WIEDEMANN_PARALLEL_IMPL

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
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
static constexpr Int kCompilePrime = 100003;

static int clamp_worker_count(int requested_threads) {
#ifdef _OPENMP
  // When OpenMP is available, respect runtime max threads unless an explicit
  // request is made.
  int hw_threads = omp_get_max_threads();
#else
  // Fallback path: no OpenMP support compiled in, so only one worker is valid.
  int hw_threads = 1;
#endif
  if (hw_threads <= 0)
    hw_threads = 1;
  if (requested_threads <= 0)
    return hw_threads;
  return std::max(1, requested_threads);
}

// Sparse matrix-vector multiply y = A*x (mod p), parallelized over rows.
// Matrix is stored as list-of-rows, where each row is (col, value) tuples.
static std::vector<Int> apply_sparse_matrix_parallel(const SparsePairMatrix &a,
                                                     const std::vector<Int> &x,
                                                     Int p, int threads) {
  (void)p;
  const int n = static_cast<int>(a.size());
  std::vector<Int> y(n, 0);

#ifdef _OPENMP
  // Explicit contiguous block partition:
  // thread t handles rows [t*chunk, min(n, (t+1)*chunk)).
#pragma omp parallel num_threads(threads)
  {
    const int actual_threads = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    const int chunk = (n + actual_threads - 1) / actual_threads;
    const int begin = tid * chunk;
    const int end = std::min(n, begin + chunk);
    for (int i = begin; i < end; ++i) {
      Int acc = 0;
      const auto &row = a[i];
      const auto *row_ptr = row.data();
      const int m = static_cast<int>(row.size());
      int t = 0;
      for (; t + 1 < m; t += 2) {
        acc += row_ptr[t].second * x[row_ptr[t].first];
        acc += row_ptr[t + 1].second * x[row_ptr[t + 1].first];
      }
      for (; t < m; ++t) {
        acc += row_ptr[t].second * x[row_ptr[t].first];
      }
      y[i] = acc % kCompilePrime;
    }
  }
#else
  (void)threads;
  for (int i = 0; i < n; ++i) {
    Int acc = 0;
    const auto &row = a[i];
    const auto *row_ptr = row.data();
    const int m = static_cast<int>(row.size());
    int t = 0;
    for (; t + 1 < m; t += 2) {
      acc += row_ptr[t].second * x[row_ptr[t].first];
      acc += row_ptr[t + 1].second * x[row_ptr[t + 1].first];
    }
    for (; t < m; ++t) {
      acc += row_ptr[t].second * x[row_ptr[t].first];
    }
    y[i] = acc % kCompilePrime;
  }
#endif

  return y;
}

// Build and cache the transpose AT in the same list-of-rows sparse format.
// This avoids re-scanning A each time we need an AT*x multiply.
static SparsePairMatrix
transpose_sparse_matrix_parallel(const SparsePairMatrix &a) {
  const int n = static_cast<int>(a.size());
  SparsePairMatrix at(n);
  std::vector<int> counts(n, 0);
  for (int i = 0; i < n; ++i) {
    for (const auto &[j, _] : a[i])
      ++counts[j];
  }
  for (int j = 0; j < n; ++j)
    at[j].reserve(counts[j]);
  for (int i = 0; i < n; ++i) {
    for (const auto &[j, v] : a[i])
      at[j].push_back({i, v});
  }
  return at;
}

// Parallel modular dot product <u,w> mod p.
// Used to generate the Krylov sequence values in Wiedemann.
static Int dot_mod_parallel(const std::vector<Int> &u,
                            const std::vector<Int> &w, Int p, int threads) {
  (void)p;
  const int n = static_cast<int>(u.size());
#ifdef _OPENMP
  std::vector<Int> partial(std::max(1, threads), 0);
#pragma omp parallel num_threads(threads)
  {
    int tid = omp_get_thread_num();
    Int local = 0;
#pragma omp for schedule(static)
    for (int i = 0; i < n; ++i) {
      local += u[i] * w[i];
    }
    partial[tid] = local % kCompilePrime;
  }

  Int sum = 0;
  for (Int v : partial)
    sum += v;
  return sum % kCompilePrime;
#else
  (void)threads;
  Int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += u[i] * w[i];
  }
  return sum % kCompilePrime;
#endif
}

// Apply preconditioned Gram operator:
//   x -> D2 * AT * D1 * A * D2 * x  (all mod p)
// where D1,D2 are random nonzero diagonal scalings.
// This is the black-box operator used by probabilistic Wiedemann rank.
static std::vector<Int> apply_preconditioned_gram_parallel(
    const SparsePairMatrix &a, const SparsePairMatrix &at,
    const std::vector<Int> &x, const std::vector<Int> &d1,
    const std::vector<Int> &d2, Int p, int threads) {
  const int n = static_cast<int>(a.size());
  std::vector<Int> tmp1(n, 0);
  std::vector<Int> tmp2(n, 0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    // First diagonal scaling by D2.
    tmp1[i] = d2[i] * x[i];
  }

  // Sparse SpMV by A.
  tmp2 = apply_sparse_matrix_parallel(a, tmp1, p, threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    // Middle diagonal scaling by D1.
    tmp2[i] = d1[i] * tmp2[i];
  }

  // Sparse SpMV by AT (precomputed transpose).
  tmp1 = apply_sparse_matrix_parallel(at, tmp2, p, threads);

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads) schedule(static)
#endif
  for (int i = 0; i < n; ++i) {
    // Final diagonal scaling by D2.
    tmp1[i] = (d2[i] * tmp1[i]) % kCompilePrime;
  }

  return tmp1;
}

// Parallelized Wiedemann rank estimator.
//
// This version parallelizes the expensive black-box operator applications and
// inner products inside each Monte Carlo trial. With OpenMP enabled
// (-fopenmp), this delivers near-linear scaling for sufficiently large n and d.
template <typename URBG>
static int rank_probabilistic_parallel_with_rng(const SparsePairMatrix &a,
                                                Int p, URBG &rng,
                                                int repeats = 3,
                                                int requested_threads = 0) {
  if (p <= 2) {
    throw std::invalid_argument("Modulus p must be an odd prime > 2");
  }
  if (p != kCompilePrime) {
    std::cerr << "[wiedemann_parallel] alert: requested prime p=" << p
              << " but this build is fixed to compile prime " << kCompilePrime
              << ". Exiting.\n";
    throw std::invalid_argument("Prime mismatch with compile-time prime");
  }
  if (repeats <= 0) {
    throw std::invalid_argument("repeats must be positive");
  }

  const int n = static_cast<int>(a.size());
  if (n == 0)
    return 0;
  const unsigned __int128 overflow_threshold =
      (static_cast<unsigned __int128>(1) << 64);
  const unsigned __int128 worst_case = static_cast<unsigned __int128>(n) *
                                       static_cast<unsigned __int128>(p) *
                                       static_cast<unsigned __int128>(p);
  if (worst_case > overflow_threshold) {
    std::cout << "[wiedemann_parallel] alert: potential overflow risk because "
                 "n*p*p > 2^64 (n="
              << n << ", p=" << p << ")\n";
  }
  // Precompute AT once per rank call.
  const SparsePairMatrix at = transpose_sparse_matrix_parallel(a);
  const int total_steps = std::max(1, repeats * 2 * n);
  int done_steps = 0;
  int next_progress = 10;
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << "[wiedemann_parallel] dims=(" << n << "," << n
            << ") progress=0% elapsed=0s\n";

  const int threads = clamp_worker_count(requested_threads);

  std::uniform_int_distribution<Int> nz_dist(1, p - 1);
  std::uniform_int_distribution<Int> any_dist(0, p - 1);

  int best_rank = 0;
  for (int rep = 0; rep < repeats; ++rep) {
    // Fresh random diagonal preconditioners and probe vector each repeat.
    std::vector<Int> d1(n), d2(n), u(n);
    for (int i = 0; i < n; ++i) {
      d1[i] = nz_dist(rng);
      d2[i] = nz_dist(rng);
      u[i] = any_dist(rng);
    }

    std::vector<Int> sequence(2 * n, 0);
    std::vector<Int> w = u;

    for (int k = 0; k < 2 * n; ++k) {
      // Krylov sequence term s_k = <u, M^k u> where M is black-box operator.
      sequence[k] = dot_mod_parallel(u, w, p, threads);
      w = apply_preconditioned_gram_parallel(a, at, w, d1, d2, p, threads);
      ++done_steps;
      while (next_progress <= 100 &&
             done_steps * 100 >= next_progress * total_steps) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - t0).count();
        std::cout << "[wiedemann_parallel] dims=(" << n << "," << n
                  << ") progress=" << next_progress << "% elapsed=" << elapsed
                  << "s\n";
        next_progress += 10;
      }
    }

    int degree =
        sparse_wiedemann::berlekamp_massey_linear_complexity(sequence, p);
    // Rank estimate from linear complexity heuristic used in this codebase.
    int estimate = (degree == n) ? n : std::max(0, degree - 1);
    best_rank = std::max(best_rank, estimate);
    if (best_rank == n)
      break;
  }

  return best_rank;
}

int rank_probabilistic_parallel(const SparsePairMatrix &a, Int p, int repeats,
                                int requested_threads,
                                std::uint64_t seed = std::random_device{}()) {
  std::mt19937_64 rng(seed);
  return rank_probabilistic_parallel_with_rng(a, p, rng, repeats,
                                              requested_threads);
}

} // namespace sparse_wiedemann_parallel

#endif // SPARSE_RANK_WIEDEMANN_PARALLEL_IMPL
