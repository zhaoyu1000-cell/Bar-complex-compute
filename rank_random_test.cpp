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

static Matrix generate_random_sparse_square(int n, int max_per_row, int64 p, std::mt19937_64& rng) {
    Matrix a(n, std::vector<int64>(n, 0));
    std::uniform_int_distribution<int> count_dist(0, max_per_row);
    std::uniform_int_distribution<int> col_dist(0, n - 1);
    std::uniform_int_distribution<int64> val_dist(1, p - 1);
    for (int i = 0; i < n; ++i) {
        int cnt = count_dist(rng);
        std::map<int, int64> row_entries;
        while (static_cast<int>(row_entries.size()) < cnt) {
            row_entries[col_dist(rng)] = val_dist(rng);
        }
        for (const auto& [c, v] : row_entries) a[i][c] = v;
    }
    return a;
}

int main() {
    constexpr int N = 100;
    constexpr int M = 1000;
    constexpr int TRIALS = 25;
    constexpr int64 P = 1000003;  // prime modulus

    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::uniform_int_distribution<int> rank_dist(0, N);

    for (int t = 1; t <= TRIALS; ++t) {
        int r = rank_dist(rng);
        Matrix matrix = generate_rank_r_matrix(N, M, r, P, rng);
        int computed_rank = rank_mod_p(matrix, P);

        if (computed_rank != r) {
            std::cerr << "Trial " << t << " failed: expected rank " << r
                      << ", got " << computed_rank << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "Trial " << t << " passed (rank = " << r << ")\n";
    }

    for (int t = 1; t <= TRIALS; ++t) {
        Matrix sparse_square = generate_random_sparse_square(N, 10, P, rng);
        int exact_rank = rank_mod_p(sparse_square, P);
        SparsePairMatrix sparse = dense_to_sparse_square(sparse_square, P);
        int wiedemann_rank = sparse_wiedemann::rank_probabilistic(sparse, P, rng, 5);

        if (wiedemann_rank != exact_rank) {
            std::cerr << "Wiedemann sparse trial " << t << " failed: expected rank " << exact_rank
                      << ", got " << wiedemann_rank << '\n';
            return EXIT_FAILURE;
        }

        std::cout << "Wiedemann sparse trial " << t << " passed (rank = " << exact_rank << ")\n";
    }

    std::cout << "All random-rank tests passed.\n";
    return EXIT_SUCCESS;
}
