#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sparse_rank_wiedemann_parallel.cpp"

using int64 = long long;
using SparseRow = std::map<int, int64>;
using SparseMatrix = std::vector<SparseRow>;
using SparsePairMatrix = sparse_wiedemann::SparsePairMatrix;

static SparseMatrix generate_sparse_full_rank_matrix(int n, int max_per_row, int64 p, std::mt19937_64& rng) {
    SparseMatrix triangular;
    triangular.reserve(n);

    std::uniform_int_distribution<int> nz_count_dist(1, max_per_row);
    std::uniform_int_distribution<int64> val_dist(1, p - 1);

    for (int i = 0; i < n; ++i) {
        SparseRow row;
        row[i] = 1;

        int max_available = i + 1;
        int target_nz = std::min(nz_count_dist(rng), max_available);
        std::unordered_set<int> used_cols;
        used_cols.reserve(static_cast<size_t>(target_nz) * 2);
        used_cols.insert(i);

        std::uniform_int_distribution<int> col_dist(0, i);
        while (static_cast<int>(row.size()) < target_nz) {
            int c = col_dist(rng);
            if (used_cols.count(c)) continue;
            used_cols.insert(c);
            row[c] = val_dist(rng);
        }

        triangular.push_back(std::move(row));
    }

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
        for (int i = 0; i < n; ++i) col_perm[i] = (i + 1) % n;
        rebuild_with_perms();
    }

    return matrix;
}

static SparsePairMatrix map_sparse_to_pair_sparse(const SparseMatrix& matrix) {
    const int n = static_cast<int>(matrix.size());
    SparsePairMatrix out(n);
    for (int i = 0; i < n; ++i) {
        out[i].reserve(matrix[i].size());
        for (const auto& [c, v] : matrix[i]) {
            out[i].push_back({c, v});
        }
    }
    return out;
}

int main(int argc, char** argv) {
    constexpr int K = 4;
    constexpr int N = 10000;                 // 10^4
    constexpr int MAX_NONZERO_PER_ROW = 10;
    constexpr int64 P = 1000003;             // prime
    constexpr int WIEDEMANN_REPEATS = 5;

    int trials = 1;
    if (argc >= 2) {
        trials = std::atoi(argv[1]);
    }
    if (trials < 1) {
        std::cerr << "Usage: " << argv[0] << " [trials>=1]\n";
        return EXIT_FAILURE;
    }

    std::vector<int> thread_counts(4);
    std::iota(thread_counts.begin(), thread_counts.end(), 1);

    std::random_device rd;
    std::mt19937_64 rng(rd());

    std::cout << "Parallel Wiedemann-only test: k=" << K << ", size=" << N << "x" << N
              << ", repeats=" << WIEDEMANN_REPEATS << "\n";

    for (int t = 1; t <= trials; ++t) {
        auto gen0 = std::chrono::steady_clock::now();
        SparseMatrix mat = generate_sparse_full_rank_matrix(N, MAX_NONZERO_PER_ROW, P, rng);
        SparsePairMatrix pair_sparse = map_sparse_to_pair_sparse(mat);
        auto gen1 = std::chrono::steady_clock::now();

        long long gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(gen1 - gen0).count();
        std::cout << "Trial " << t << " matrix generated in " << gen_ms << " ms\n";

        for (int threads : thread_counts) {
            auto t0 = std::chrono::steady_clock::now();
            int rank = sparse_wiedemann_parallel::rank_probabilistic_parallel(
                pair_sparse, P, WIEDEMANN_REPEATS, threads, static_cast<std::uint64_t>(rng()));
            auto t1 = std::chrono::steady_clock::now();
            long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

            if (rank != N) {
                std::cerr << "  threads=" << threads << " FAILED: expected rank " << N
                          << ", got " << rank << " (" << ms << " ms)\n";
                return EXIT_FAILURE;
            }

            std::cout << "  threads=" << threads << " passed (rank=" << rank
                      << ", time=" << ms << " ms)\n";
        }
    }

    std::cout << "All parallel Wiedemann k=4 tests passed.\n";
    return EXIT_SUCCESS;
}
