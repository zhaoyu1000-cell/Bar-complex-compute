#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sparse_rank_wiedemann.cpp"

using int64 = long long;
using SparseRow = std::map<int, int64>;  // sorted keys => quick lead-column lookup
using SparseMatrix = std::vector<SparseRow>;
using SparsePairMatrix = sparse_wiedemann::SparsePairMatrix;

static int64 normalize_mod(int64 x, int64 p) {
    x %= p;
    if (x < 0) x += p;
    return x;
}

static int64 mod_pow(int64 base, int64 exp, int64 mod) {
    int64 result = 1;
    base = normalize_mod(base, mod);
    while (exp > 0) {
        if (exp & 1LL) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1LL;
    }
    return result;
}

static int64 mod_inverse(int64 a, int64 p) {
    a = normalize_mod(a, p);
    if (a == 0) {
        throw std::invalid_argument("Zero has no modular inverse");
    }
    return mod_pow(a, p - 2, p);  // p must be prime
}

static void scale_row(SparseRow& row, int64 scalar, int64 p) {
    if (scalar == 1) return;
    for (auto it = row.begin(); it != row.end();) {
        it->second = normalize_mod(it->second * scalar, p);
        if (it->second == 0) {
            it = row.erase(it);
        } else {
            ++it;
        }
    }
}

static void add_scaled_row(SparseRow& target, const SparseRow& src, int64 scale, int64 p) {
    if (scale == 0) return;
    for (const auto& [col, value] : src) {
        int64 updated = normalize_mod(target[col] + scale * value, p);
        if (updated == 0) {
            target.erase(col);
        } else {
            target[col] = updated;
        }
    }
}

// Sparse rank computation via incremental elimination into a pivot basis.
static int sparse_rank_mod_p(const SparseMatrix& matrix, int64 p, int ncols) {
    std::vector<SparseRow> pivot_basis(ncols);
    std::vector<char> has_pivot(ncols, 0);
    int rank = 0;

    for (const auto& input_row : matrix) {
        SparseRow row;
        for (const auto& [col, value] : input_row) {
            int64 v = normalize_mod(value, p);
            if (v != 0) {
                row[col] = v;
            }
        }

        while (!row.empty()) {
            int lead_col = row.begin()->first;
            int64 lead_val = row.begin()->second;

            if (!has_pivot[lead_col]) {
                int64 inv = mod_inverse(lead_val, p);
                scale_row(row, inv, p);  // make pivot coefficient 1
                pivot_basis[lead_col] = std::move(row);
                has_pivot[lead_col] = 1;
                ++rank;
                break;
            }

            // Eliminate using existing pivot row for lead_col.
            int64 factor = normalize_mod(-lead_val, p);
            add_scaled_row(row, pivot_basis[lead_col], factor, p);
        }
    }

    return rank;
}

static SparseMatrix generate_sparse_full_rank_matrix(int n, int max_per_row, int64 p, std::mt19937_64& rng) {
    SparseMatrix triangular;
    triangular.reserve(n);

    std::uniform_int_distribution<int> nz_count_dist(1, max_per_row);
    std::uniform_int_distribution<int64> val_dist(1, p - 1);

    for (int i = 0; i < n; ++i) {
        SparseRow row;

        // Build a unit lower-triangular sparse matrix (full rank by construction).
        row[i] = 1;

        int max_available = i + 1;  // columns 0..i
        int target_nz = std::min(nz_count_dist(rng), max_available);
        std::unordered_set<int> used_cols;
        used_cols.reserve(static_cast<size_t>(target_nz) * 2);
        used_cols.insert(i);

        // Keep random non-zeros on/below diagonal while preserving sparsity.
        std::uniform_int_distribution<int> col_dist(0, i);
        while (static_cast<int>(row.size()) < target_nz) {
            int c = col_dist(rng);
            if (used_cols.count(c)) continue;
            used_cols.insert(c);
            row[c] = val_dist(rng);
        }

        triangular.push_back(std::move(row));
    }

    // Randomly permute rows and columns so the final matrix is *not* triangular.
    std::vector<int> row_perm(n), col_perm(n);
    for (int i = 0; i < n; ++i) {
        row_perm[i] = i;
        col_perm[i] = i;
    }

    SparseMatrix matrix;
    auto rebuild_with_perms = [&]() {
        matrix.assign(n, SparseRow{});
        for (int new_r = 0; new_r < n; ++new_r) {
            int old_r = row_perm[new_r];
            for (const auto& [old_c, value] : triangular[old_r]) {
                matrix[new_r][col_perm[old_c]] = value;
            }
        }
    };

    bool has_above = false;
    bool has_below = false;
    for (int attempt = 0; attempt < 16; ++attempt) {
        std::shuffle(row_perm.begin(), row_perm.end(), rng);
        std::shuffle(col_perm.begin(), col_perm.end(), rng);
        rebuild_with_perms();

        has_above = false;
        has_below = false;
        for (int r = 0; r < n && !(has_above && has_below); ++r) {
            for (const auto& [c, _] : matrix[r]) {
                if (c > r) has_above = true;
                if (c < r) has_below = true;
                if (has_above && has_below) break;
            }
        }
        if (has_above && has_below) break;
    }

    if (!(has_above && has_below) && n >= 2) {
        // Deterministic fallback: keep rows, rotate columns by one.
        for (int i = 0; i < n; ++i) col_perm[i] = (i + 1) % n;
        rebuild_with_perms();
    }

    return matrix;
}

static SparsePairMatrix map_sparse_to_pair_sparse(const SparseMatrix& matrix, int64 p) {
    const int n = static_cast<int>(matrix.size());
    SparsePairMatrix out(n);
    for (int i = 0; i < n; ++i) {
        for (const auto& [c, v] : matrix[i]) {
            int64 nv = normalize_mod(v, p);
            if (nv != 0) out[i].push_back({c, nv});
        }
    }
    return out;
}

int main(int argc, char** argv) {
    constexpr int MAX_NONZERO_PER_ROW = 10;
    constexpr int64 P = 1000003;  // prime
    int max_k = 4;
    int trials_per_k = 3;
    int max_wiedemann_n = 1000;

    if (argc >= 2) {
        max_k = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        trials_per_k = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        max_wiedemann_n = std::atoi(argv[3]);
    }
    if (max_k < 1 || trials_per_k < 1 || max_wiedemann_n < 0) {
        std::cerr << "Usage: " << argv[0]
                  << " [max_k>=1] [trials_per_k>=1] [max_wiedemann_n>=0]\n";
        return EXIT_FAILURE;
    }

    std::random_device rd;
    std::mt19937_64 rng(rd());

    for (int k = 1; k <= max_k; ++k) {
        int n = 1;
        for (int i = 0; i < k; ++i) n *= 10;

        std::cout << "k=" << k << ", size=" << n << "x" << n << "\n";

        for (int t = 1; t <= trials_per_k; ++t) {
            auto t0 = std::chrono::steady_clock::now();
            SparseMatrix mat = generate_sparse_full_rank_matrix(n, MAX_NONZERO_PER_ROW, P, rng);
            auto t1 = std::chrono::steady_clock::now();
            int rank = sparse_rank_mod_p(mat, P, n);
            auto t2 = std::chrono::steady_clock::now();

            auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            auto elim_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            if (rank != n) {
                std::cerr << "  Trial " << t << " FAILED: expected rank " << n << ", got " << rank
                          << " (gen " << gen_ms << " ms, elim " << elim_ms << " ms)\n";
                return EXIT_FAILURE;
            }

            // Black-box rank check (Wiedemann) is O(n^2 d), so cap n to keep runtime practical.
            bool ran_wiedemann = false;
            long long wiedemann_ms = 0;
            if (n <= max_wiedemann_n) {
                auto tw0 = std::chrono::steady_clock::now();
                SparsePairMatrix pair_sparse = map_sparse_to_pair_sparse(mat, P);
                int w_rank = sparse_wiedemann::rank_probabilistic(pair_sparse, P, rng, 5);
                auto tw1 = std::chrono::steady_clock::now();
                wiedemann_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tw1 - tw0).count();
                ran_wiedemann = true;
                if (w_rank != n) {
                    std::cerr << "  Trial " << t << " Wiedemann FAILED: expected rank " << n << ", got "
                              << w_rank << " (gen " << gen_ms << " ms, elim " << elim_ms
                              << " ms, wiedemann " << wiedemann_ms << " ms)\n";
                    return EXIT_FAILURE;
                }
            }

            std::cout << "  Trial " << t << " passed (rank=" << rank << ", gen=" << gen_ms
                      << " ms, elim=" << elim_ms << " ms";
            if (ran_wiedemann) {
                std::cout << ", wiedemann=" << wiedemann_ms << " ms";
            } else {
                std::cout << ", wiedemann=skipped";
            }
            std::cout << ")\n";
        }
    }

    std::cout << "All sparse large-matrix tests passed up to k=" << max_k << ".\n";
    return EXIT_SUCCESS;
}
