#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sparse_wiedemann {

using Int = long long;
using SparsePairRow = std::vector<std::pair<int, Int>>;
using SparsePairMatrix = std::vector<SparsePairRow>;

static Int normalize_mod(Int x, Int p) {
    x %= p;
    if (x < 0) x += p;
    return x;
}

static std::vector<Int> apply_sparse_matrix(const SparsePairMatrix& a, const std::vector<Int>& x, Int p) {
    const int n = static_cast<int>(a.size());
    std::vector<Int> y(n, 0);
    for (int i = 0; i < n; ++i) {
        Int acc = 0;
        for (const auto& [j, val] : a[i]) {
            acc = (acc + val * x[j]) % p;
        }
        y[i] = acc;
    }
    return y;
}

static std::vector<Int> apply_sparse_matrix_transpose(const SparsePairMatrix& a, const std::vector<Int>& x, Int p) {
    const int n = static_cast<int>(a.size());
    std::vector<Int> y(n, 0);
    for (int i = 0; i < n; ++i) {
        const Int xi = x[i];
        if (xi == 0) continue;
        for (const auto& [j, val] : a[i]) {
            y[j] = (y[j] + val * xi) % p;
        }
    }
    return y;
}

static int berlekamp_massey_linear_complexity(const std::vector<Int>& sequence, Int p) {
    std::vector<Int> c(1, 1), b(1, 1);
    int l = 0;
    int m = 1;
    Int bb = 1;

    auto mod_pow = [&](Int base, Int exp) {
        Int result = 1;
        base = normalize_mod(base, p);
        while (exp > 0) {
            if (exp & 1LL) result = (result * base) % p;
            base = (base * base) % p;
            exp >>= 1LL;
        }
        return result;
    };

    auto mod_inv = [&](Int x) {
        x = normalize_mod(x, p);
        if (x == 0) throw std::invalid_argument("Berlekamp-Massey division by zero");
        return mod_pow(x, p - 2);
    };

    for (int n = 0; n < static_cast<int>(sequence.size()); ++n) {
        Int d = sequence[n];
        for (int i = 1; i <= l; ++i) {
            d = normalize_mod(d + c[i] * sequence[n - i], p);
        }
        if (d == 0) {
            ++m;
            continue;
        }

        std::vector<Int> t = c;
        Int coef = normalize_mod(d * mod_inv(bb), p);
        if (static_cast<int>(c.size()) < static_cast<int>(b.size()) + m) {
            c.resize(static_cast<int>(b.size()) + m, 0);
        }
        for (int i = 0; i < static_cast<int>(b.size()); ++i) {
            c[i + m] = normalize_mod(c[i + m] - coef * b[i], p);
        }

        if (2 * l <= n) {
            l = n + 1 - l;
            b = std::move(t);
            bb = d;
            m = 1;
        } else {
            ++m;
        }
    }

    return l;
}

template <typename URBG>
int rank_probabilistic(const SparsePairMatrix& a, Int p, URBG& rng, int repeats = 3) {
    const int n = static_cast<int>(a.size());
    if (n == 0) return 0;

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

        auto apply_preconditioned_gram = [&](const std::vector<Int>& x) {
            std::vector<Int> t(n);
            for (int i = 0; i < n; ++i) t[i] = (d2[i] * x[i]) % p;
            t = apply_sparse_matrix(a, t, p);
            for (int i = 0; i < n; ++i) t[i] = (d1[i] * t[i]) % p;
            t = apply_sparse_matrix_transpose(a, t, p);
            for (int i = 0; i < n; ++i) t[i] = (d2[i] * t[i]) % p;
            return t;
        };

        std::vector<Int> sequence(2 * n, 0);
        std::vector<Int> w = u;
        for (int k = 0; k < 2 * n; ++k) {
            Int sk = 0;
            for (int i = 0; i < n; ++i) sk = (sk + u[i] * w[i]) % p;
            sequence[k] = sk;
            w = apply_preconditioned_gram(w);
        }

        int degree = berlekamp_massey_linear_complexity(sequence, p);
        int estimate = (degree == n) ? n : std::max(0, degree - 1);
        best_rank = std::max(best_rank, estimate);
        if (best_rank == n) break;
    }

    return best_rank;
}

}  // namespace sparse_wiedemann
