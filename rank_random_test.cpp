#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <utility>
#include <vector>

#define MATRIX_RANK_FP_NO_MAIN
#include "matrix_rank_fp.cpp"
#include "sparse_rank_wiedemann.cpp"

using int64 = long long;
using Matrix = std::vector<std::vector<int64>>;
using SparsePairMatrix = sparse_wiedemann::SparsePairMatrix;

static Matrix generate_full_row_rank_basis(int r, int m, int64 p, std::mt19937_64& rng) {
    std::uniform_int_distribution<int64> entry_dist(0, p - 1);

    Matrix basis(r, std::vector<int64>(m));
    if (r == 0) {
        return basis;
    }

    while (true) {
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < m; ++j) {
                basis[i][j] = entry_dist(rng);
            }
        }

        if (rank_mod_p(basis, p) == r) {
            return basis;
        }
    }
}

static Matrix generate_rank_r_matrix(int n, int m, int r, int64 p, std::mt19937_64& rng) {
    Matrix matrix(n, std::vector<int64>(m, 0));
    if (r == 0) {
        return matrix;
    }

    Matrix basis = generate_full_row_rank_basis(r, m, p, rng);
    std::uniform_int_distribution<int64> coeff_dist(0, p - 1);

    // Guarantee row-rank >= r.
    for (int i = 0; i < r && i < n; ++i) {
        matrix[i] = basis[i];
    }

    // Remaining rows are random linear combinations of the basis vectors.
    for (int i = r; i < n; ++i) {
        std::vector<int64> coeffs(r);
        for (int k = 0; k < r; ++k) {
            coeffs[k] = coeff_dist(rng);
        }

        for (int j = 0; j < m; ++j) {
            int64 value = 0;
            for (int k = 0; k < r; ++k) {
                value = (value + coeffs[k] * basis[k][j]) % p;
            }
            matrix[i][j] = value;
        }
    }

    return matrix;
}

static SparsePairMatrix dense_to_sparse_square(const Matrix& a, int64 p) {
    int n = static_cast<int>(a.size());
    SparsePairMatrix sparse(n);
    for (int i = 0; i < n; ++i) {
        int m = static_cast<int>(a[i].size());
        for (int j = 0; j < m; ++j) {
            int64 v = normalize_mod(a[i][j], p);
            if (v != 0) sparse[i].push_back({j, v});
        }
    }
    return sparse;
}

static Matrix generate_sparse_rank_r_square(int n,
                                            int r,
                                            int max_per_basis_row,
                                            int64 p,
                                            std::mt19937_64& rng) {
    if (r < 0 || r > n) {
        throw std::invalid_argument("generate_sparse_rank_r_square: rank out of range");
    }
    Matrix a(n, std::vector<int64>(n, 0));
    if (r == 0) return a;

    std::vector<int> pivot_cols(n);
    for (int i = 0; i < n; ++i) pivot_cols[i] = i;
    std::shuffle(pivot_cols.begin(), pivot_cols.end(), rng);
    pivot_cols.resize(r);

    std::uniform_int_distribution<int64> nz_val_dist(1, p - 1);
    std::uniform_int_distribution<int> extra_dist(0, std::max(0, max_per_basis_row - 1));
    std::uniform_int_distribution<int> col_dist(0, n - 1);

    // Build r sparse independent basis rows with unique pivot columns.
    for (int i = 0; i < r; ++i) {
        std::map<int, int64> entries;
        entries[pivot_cols[i]] = 1;
        const int extras = extra_dist(rng);
        while (static_cast<int>(entries.size()) < 1 + extras) {
            int c = col_dist(rng);
            if (entries.count(c)) continue;
            entries[c] = nz_val_dist(rng);
        }
        for (const auto& [c, v] : entries) a[i][c] = v;
    }

    // Dependent sparse rows: random linear combinations of two basis rows.
    if (r > 0) {
        std::uniform_int_distribution<int> basis_dist(0, r - 1);
        for (int i = r; i < n; ++i) {
            int b1 = basis_dist(rng);
            int b2 = basis_dist(rng);
            int64 c1 = nz_val_dist(rng);
            int64 c2 = nz_val_dist(rng);
            for (int j = 0; j < n; ++j) {
                int64 v = (c1 * a[b1][j] + c2 * a[b2][j]) % p;
                a[i][j] = v;
            }
        }
    }

    return a;
}

int main(int argc, char** argv) {
    constexpr int N = 100;
    constexpr int M = 1000;
    constexpr int64 P = 1000003;  // prime modulus
    constexpr int SPARSE_TARGET_RANK = N / 2;
    constexpr int SPARSE_MAX_PER_BASIS_ROW = 10;

    int dense_trials = 25;
    int sparse_trials = 25;
    int max_repeats = 5;
    if (argc >= 2) dense_trials = std::atoi(argv[1]);
    if (argc >= 3) sparse_trials = std::atoi(argv[2]);
    if (argc >= 4) max_repeats = std::atoi(argv[3]);
    if (dense_trials < 1 || sparse_trials < 1 || max_repeats < 1) {
        std::cerr << "Usage: " << argv[0]
                  << " [dense_trials>=1] [sparse_trials>=1] [max_repeats>=1]\n";
        return EXIT_FAILURE;
    }

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<int> rank_dist(0, N);

    int dense_failures = 0;
    for (int t = 1; t <= dense_trials; ++t) {
        int r = rank_dist(rng);
        Matrix matrix = generate_rank_r_matrix(N, M, r, P, rng);
        int computed_rank = rank_mod_p(matrix, P);

        if (computed_rank != r) {
            ++dense_failures;
        }
    }
    std::cout << "Dense failures: " << dense_failures << " / " << dense_trials << "\n";
    if (dense_failures > 0) {
        std::cerr << "Dense test failures detected.\n";
        return EXIT_FAILURE;
    }

    for (int repeats = 1; repeats <= max_repeats; ++repeats) {
        int sparse_failures = 0;
        for (int t = 1; t <= sparse_trials; ++t) {
            Matrix sparse_square = generate_sparse_rank_r_square(
                N, SPARSE_TARGET_RANK, SPARSE_MAX_PER_BASIS_ROW, P, rng);
            int exact_rank = rank_mod_p(sparse_square, P);
            SparsePairMatrix sparse = dense_to_sparse_square(sparse_square, P);
            int wiedemann_rank = sparse_wiedemann::rank_probabilistic(sparse, P, rng, repeats);

            if (wiedemann_rank != exact_rank) {
                ++sparse_failures;
            }
        }
        std::cout << "Sparse failures (target rank " << SPARSE_TARGET_RANK
                  << "), repeats=" << repeats << ": " << sparse_failures << " / " << sparse_trials
                  << "\n";
    }

    std::cout << "Repeat sweep complete.\n";
    return EXIT_SUCCESS;
}
