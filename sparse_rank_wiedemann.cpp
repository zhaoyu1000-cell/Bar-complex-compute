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

static inline Int normalize_mod(Int x, Int p) {
    x %= p;
    if (x < 0) x += p;
    return x;
}

struct CSRMatrix {
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col;
    std::vector<Int> val;
};

static CSRMatrix build_csr(const SparsePairMatrix& a, Int p) {
    const int n = static_cast<int>(a.size());
    CSRMatrix m;
    m.n = n;
    m.row_ptr.resize(n + 1, 0);

    std::size_t nnz = 0;
    for (int i = 0; i < n; ++i) nnz += a[i].size();
    m.col.reserve(nnz);
    m.val.reserve(nnz);

    for (int i = 0; i < n; ++i) {
        m.row_ptr[i] = static_cast<int>(m.col.size());
        for (const auto& [j, v] : a[i]) {
            Int vv = normalize_mod(v, p);
            if (vv == 0) continue;
            m.col.push_back(j);
            m.val.push_back(vv);
        }
    }
    m.row_ptr[n] = static_cast<int>(m.col.size());
    return m;
}

static CSRMatrix build_transpose_csr(const SparsePairMatrix& a, Int p) {
    const int n = static_cast<int>(a.size());
    std::vector<int> deg(n, 0);
    for (int i = 0; i < n; ++i) {
        for (const auto& [j, v] : a[i]) {
            if (normalize_mod(v, p) != 0) ++deg[j];
        }
    }

    CSRMatrix t;
    t.n = n;
    t.row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) t.row_ptr[i + 1] = t.row_ptr[i] + deg[i];

    const int nnz = t.row_ptr[n];
    t.col.assign(nnz, 0);
    t.val.assign(nnz, 0);

    std::vector<int> cur = t.row_ptr;
    for (int i = 0; i < n; ++i) {
        for (const auto& [j, v] : a[i]) {
            Int vv = normalize_mod(v, p);
            if (vv == 0) continue;
            int pos = cur[j]++;
            t.col[pos] = i;
            t.val[pos] = vv;
        }
    }
    return t;
}

static inline void csr_matvec(const CSRMatrix& m,
                              const std::vector<Int>& x,
                              std::vector<Int>& y,
                              Int p) {
    const int n = m.n;
    for (int i = 0; i < n; ++i) {
        Int acc = 0;
        for (int e = m.row_ptr[i]; e < m.row_ptr[i + 1]; ++e) {
            acc += m.val[e] * x[m.col[e]];
        }
        y[i] = acc % p;
    }
}

static inline Int dot_mod(const std::vector<Int>& a,
                          const std::vector<Int>& b,
                          Int p) {
    Int acc = 0;
    const int n = static_cast<int>(a.size());
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc % p;
}

static inline Int mod_pow(Int base, Int exp, Int p) {
    Int result = 1;
    base = normalize_mod(base, p);
    while (exp > 0) {
        if (exp & 1LL) result = (result * base) % p;
        base = (base * base) % p;
        exp >>= 1LL;
    }
    return result;
}

static inline Int mod_inv(Int x, Int p) {
    x = normalize_mod(x, p);
    if (x == 0) throw std::invalid_argument("Berlekamp-Massey division by zero");
    return mod_pow(x, p - 2, p);
}

static int berlekamp_massey_linear_complexity(const std::vector<Int>& sequence, Int p) {
    std::vector<Int> c(1, 1), b(1, 1);
    int l = 0;
    int m = 1;
    Int bb = 1;

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
        Int coef = normalize_mod(d * mod_inv(bb, p), p);
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
int rank_probabilistic(const SparsePairMatrix& a, Int p, URBG& rng, int repeats = 1) {
    const int n = static_cast<int>(a.size());
    if (n == 0) return 0;
    if (p <= 2) throw std::invalid_argument("Modulus p must be an odd prime > 2");

    const CSRMatrix csr = build_csr(a, p);
    const CSRMatrix cst = build_transpose_csr(a, p);

    std::uniform_int_distribution<Int> nz_dist(1, p - 1);
    std::uniform_int_distribution<Int> any_dist(0, p - 1);

    std::vector<Int> d1(n), d2(n), u(n), w(n), t1(n), t2(n), sequence(2 * n);

    int best_rank = 0;

    for (int rep = 0; rep < repeats; ++rep) {
        for (int i = 0; i < n; ++i) {
            d1[i] = nz_dist(rng);
            d2[i] = nz_dist(rng);
            u[i] = any_dist(rng);
            w[i] = u[i];
        }

        for (int k = 0; k < 2 * n; ++k) {
            sequence[k] = dot_mod(u, w, p);

            for (int i = 0; i < n; ++i) t1[i] = (d2[i] * w[i]) % p;
            csr_matvec(csr, t1, t2, p);
            for (int i = 0; i < n; ++i) t2[i] = (d1[i] * t2[i]) % p;
            csr_matvec(cst, t2, t1, p);
            for (int i = 0; i < n; ++i) w[i] = (d2[i] * t1[i]) % p;
        }

        int degree = berlekamp_massey_linear_complexity(sequence, p);
        int estimate = (degree == n) ? n : std::max(0, degree - 1);
        if (estimate > best_rank) best_rank = estimate;
        if (best_rank == n) break;
    }

    return best_rank;
}

}  // namespace sparse_wiedemann
