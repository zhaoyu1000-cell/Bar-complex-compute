#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparse_rank_wiedemann_parallel.cpp"

namespace {

using Int = long long;
constexpr Int MOD = 1000003;  // odd prime for Wiedemann
constexpr int MAX_N = 4;

struct BasisSpace {
    int n = 0;
    int word_count = 0;
    std::vector<std::vector<Int>> basis_vectors;  // coordinates in T(V)_n standard basis
    std::unordered_map<std::string, int> pivot_row_by_key;
};

struct RankTiming {
    int rows = 0;
    int cols = 0;
    int rank = 0;
    long long ms = 0;
};

static Int mod_norm(Int x) {
    x %= MOD;
    if (x < 0) x += MOD;
    return x;
}

static Int mod_pow(Int a, Int e) {
    Int r = 1;
    a = mod_norm(a);
    while (e > 0) {
        if (e & 1LL) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1LL;
    }
    return r;
}

static Int mod_inv(Int x) {
    return mod_pow(mod_norm(x), MOD - 2);
}

static int encode_word(const std::vector<int>& w) {
    int id = 0;
    for (int x : w) id = id * 3 + x;
    return id;
}

static std::vector<int> decode_word(int id, int n) {
    std::vector<int> w(n, 0);
    for (int i = n - 1; i >= 0; --i) {
        w[i] = id % 3;
        id /= 3;
    }
    return w;
}

static std::vector<int> pow3(MAX_N + 1, 1);

// Rack: conjugation on transpositions of S_3.
// labels: 0=(12), 1=(13), 2=(23)
static const int rack_table[3][3] = {
    {0, 2, 1},
    {2, 1, 0},
    {1, 0, 2},
};

static inline int rack_op(int a, int b) {
    return rack_table[a][b];
}

static std::vector<std::vector<int>> compositions(int n, int k) {
    std::vector<std::vector<int>> out;
    std::vector<int> cur;
    std::function<void(int, int)> dfs = [&](int rem, int parts) {
        if (parts == 1) {
            cur.push_back(rem);
            out.push_back(cur);
            cur.pop_back();
            return;
        }
        for (int x = 1; x <= rem - parts + 1; ++x) {
            cur.push_back(x);
            dfs(rem - x, parts - 1);
            cur.pop_back();
        }
    };
    dfs(n, k);
    return out;
}

// Build shuffle permutations as list p where p[new_pos]=old_pos.
static std::vector<std::vector<int>> shuffles(int r, int s) {
    std::vector<std::vector<int>> out;
    std::vector<int> pick(r + s, 0);
    for (int i = 0; i < r; ++i) pick[i] = 1;
    std::sort(pick.begin(), pick.end());
    do {
        std::vector<int> p;
        p.reserve(r + s);
        int a = 0, b = r;
        for (int bit : pick) {
            if (bit == 0) p.push_back(b++);
            else p.push_back(a++);
        }
        out.push_back(std::move(p));
    } while (std::next_permutation(pick.begin(), pick.end()));
    return out;
}

// Apply braid operator realization for a given shuffle permutation.
static std::pair<Int, std::vector<int>> apply_shuffle_braid(const std::vector<int>& u,
                                                             const std::vector<int>& v,
                                                             const std::vector<int>& p) {
    std::vector<int> letters = u;
    letters.insert(letters.end(), v.begin(), v.end());
    const int n = static_cast<int>(letters.size());

    std::vector<int> pos(n);
    std::iota(pos.begin(), pos.end(), 0);

    std::vector<int> target_pos(n, 0);
    for (int i = 0; i < n; ++i) target_pos[p[i]] = i;

    Int coeff = 1;
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 0; i + 1 < n; ++i) {
            if (target_pos[pos[i]] > target_pos[pos[i + 1]]) {
                changed = true;
                coeff = mod_norm(-coeff);  // cocycle q=-1

                int a = letters[i];
                int b = letters[i + 1];
                int new_left = rack_op(a, b);
                int new_right = a;
                letters[i] = new_left;
                letters[i + 1] = new_right;

                std::swap(pos[i], pos[i + 1]);
            }
        }
    }

    return {coeff, letters};
}

static std::vector<Int> qshuffle_word_product(const std::vector<int>& u, const std::vector<int>& v) {
    const int r = static_cast<int>(u.size());
    const int s = static_cast<int>(v.size());
    const int n = r + s;
    std::vector<Int> out(pow3[n], 0);

    auto sh = shuffles(r, s);
    for (const auto& p : sh) {
        auto [coef, w] = apply_shuffle_braid(u, v, p);
        int idx = encode_word(w);
        out[idx] = mod_norm(out[idx] + coef);
    }
    return out;
}

static std::vector<Int> qshuffle_product_vec(const std::vector<Int>& a,
                                             const std::vector<Int>& b,
                                             int da,
                                             int db) {
    const int dim_a = pow3[da];
    const int dim_b = pow3[db];
    const int dim_c = pow3[da + db];
    std::vector<Int> out(dim_c, 0);

    for (int i = 0; i < dim_a; ++i) {
        if (a[i] == 0) continue;
        auto wi = decode_word(i, da);
        for (int j = 0; j < dim_b; ++j) {
            if (b[j] == 0) continue;
            auto wj = decode_word(j, db);
            auto wprod = qshuffle_word_product(wi, wj);
            Int scale = (a[i] * b[j]) % MOD;
            for (int k = 0; k < dim_c; ++k) {
                if (wprod[k] == 0) continue;
                out[k] = mod_norm(out[k] + scale * wprod[k]);
            }
        }
    }
    return out;
}

// Row-reduced basis with pivot hash key
static BasisSpace make_basis_space(int n, const std::vector<std::vector<Int>>& generators) {
    BasisSpace sp;
    sp.n = n;
    sp.word_count = pow3[n];

    std::vector<std::vector<Int>> rows;
    std::vector<int> pivots;

    for (auto v : generators) {
        for (size_t r = 0; r < rows.size(); ++r) {
            int p = pivots[r];
            if (v[p] != 0) {
                Int factor = (v[p] * mod_inv(rows[r][p])) % MOD;
                for (int c = p; c < sp.word_count; ++c) {
                    v[c] = mod_norm(v[c] - factor * rows[r][c]);
                }
            }
        }

        int pivot = -1;
        for (int c = 0; c < sp.word_count; ++c) {
            if (v[c] != 0) {
                pivot = c;
                break;
            }
        }
        if (pivot < 0) continue;

        Int inv = mod_inv(v[pivot]);
        for (int c = pivot; c < sp.word_count; ++c) v[c] = (v[c] * inv) % MOD;

        for (size_t r = 0; r < rows.size(); ++r) {
            if (rows[r][pivot] != 0) {
                Int factor = rows[r][pivot];
                for (int c = pivot; c < sp.word_count; ++c) {
                    rows[r][c] = mod_norm(rows[r][c] - factor * v[c]);
                }
            }
        }

        size_t ins = 0;
        while (ins < pivots.size() && pivots[ins] < pivot) ++ins;
        rows.insert(rows.begin() + static_cast<long>(ins), v);
        pivots.insert(pivots.begin() + static_cast<long>(ins), pivot);
    }

    sp.basis_vectors = rows;
    for (size_t i = 0; i < rows.size(); ++i) {
        int p = pivots[i];
        sp.pivot_row_by_key[std::to_string(p)] = static_cast<int>(i);
    }
    return sp;
}

// Coordinates of vec in basis sp (must lie in span)
static std::vector<Int> coordinates_in_basis(const BasisSpace& sp, std::vector<Int> vec) {
    std::vector<Int> coord(sp.basis_vectors.size(), 0);
    for (size_t r = 0; r < sp.basis_vectors.size(); ++r) {
        const auto& br = sp.basis_vectors[r];
        int pivot = -1;
        for (int c = 0; c < sp.word_count; ++c) {
            if (br[c] != 0) {
                pivot = c;
                break;
            }
        }
        if (pivot < 0) continue;
        if (vec[pivot] != 0) {
            Int t = vec[pivot];
            coord[r] = t;
            for (int c = pivot; c < sp.word_count; ++c) {
                vec[c] = mod_norm(vec[c] - t * br[c]);
            }
        }
    }
    return coord;
}

static sparse_wiedemann::SparsePairMatrix dense_to_sparse_square(const std::vector<std::vector<Int>>& m) {
    int n = static_cast<int>(m.size());
    sparse_wiedemann::SparsePairMatrix sp(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Int v = mod_norm(m[i][j]);
            if (v != 0) sp[i].push_back({j, v});
        }
    }
    return sp;
}

static RankTiming rank_rectangular_wiedemann(const std::vector<std::vector<Int>>& m,
                                             int threads,
                                             int repeats,
                                             std::uint64_t seed) {
    RankTiming rt;
    rt.rows = static_cast<int>(m.size());
    rt.cols = rt.rows == 0 ? 0 : static_cast<int>(m[0].size());

    std::vector<std::vector<Int>> gram(rt.rows, std::vector<Int>(rt.rows, 0));
    for (int i = 0; i < rt.rows; ++i) {
        for (int j = i; j < rt.rows; ++j) {
            Int s = 0;
            for (int k = 0; k < rt.cols; ++k) s = (s + m[i][k] * m[j][k]) % MOD;
            gram[i][j] = gram[j][i] = s;
        }
    }

    auto sparse = dense_to_sparse_square(gram);
    auto t0 = std::chrono::steady_clock::now();
    rt.rank = sparse_wiedemann_parallel::rank_probabilistic_parallel(sparse, MOD, repeats, threads, seed);
    auto t1 = std::chrono::steady_clock::now();
    rt.ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    return rt;
}

}  // namespace

int main() {
    for (int i = 1; i <= MAX_N; ++i) pow3[i] = pow3[i - 1] * 3;

    // Build A_n as subspace of T(V)_n generated by shuffle products of degree-1 letters.
    std::vector<BasisSpace> A(MAX_N + 1);
    A[0] = make_basis_space(0, {{1}});

    std::vector<std::vector<Int>> degree1(3, std::vector<Int>(pow3[1], 0));
    for (int x = 0; x < 3; ++x) degree1[x][x] = 1;
    A[1] = make_basis_space(1, degree1);

    for (int n = 2; n <= MAX_N; ++n) {
        std::vector<std::vector<Int>> gens;
        for (const auto& v : A[n - 1].basis_vectors) {
            for (const auto& g : A[1].basis_vectors) {
                gens.push_back(qshuffle_product_vec(v, g, n - 1, 1));
                gens.push_back(qshuffle_product_vec(g, v, 1, n - 1));
            }
        }
        A[n] = make_basis_space(n, gens);
    }

    std::cout << "Degree dimensions dim(A_n):\n";
    for (int n = 0; n <= MAX_N; ++n) {
        std::cout << "  n=" << n << " -> " << A[n].basis_vectors.size() << "\n";
    }

    int threads = 8;
    int repeats = 10;
    std::mt19937_64 rng(123456789ULL);

    long long total_ms = 0;

    for (int n = 1; n <= MAX_N; ++n) {
        std::cout << "\n=== Weight n=" << n << " bar complex ===\n";

        // C_k indexed by number of parts k (1..n), with C_n=A_1^{\otimes n}, C_1=A_n
        std::vector<int> Cdim(n + 1, 0);
        for (int k = 1; k <= n; ++k) {
            auto comps = compositions(n, k);
            int dim = 0;
            for (const auto& c : comps) {
                int d = 1;
                for (int part : c) d *= static_cast<int>(A[part].basis_vectors.size());
                dim += d;
            }
            Cdim[k] = dim;
            std::cout << "  dim B_" << k << " = " << dim << "\n";
        }
        std::cout << "  Complex shape (left to right): ";
        for (int k = n; k >= 1; --k) {
            std::cout << "B_" << k << "[" << Cdim[k] << "]";
            if (k > 1) std::cout << " -> ";
        }
        std::cout << "\n";

        // Differentals d_k: C_k -> C_{k-1} for k=n..2
        std::vector<int> rank_d(n + 1, 0);
        std::vector<RankTiming> timings(n + 1);

        for (int k = n; k >= 2; --k) {
            auto comps = compositions(n, k);
            auto comps_prev = compositions(n, k - 1);

            std::map<std::vector<int>, int> block_row_offset;
            int row_off = 0;
            for (const auto& cp : comps_prev) {
                block_row_offset[cp] = row_off;
                int d = 1;
                for (int part : cp) d *= static_cast<int>(A[part].basis_vectors.size());
                row_off += d;
            }

            std::map<std::vector<int>, int> block_col_offset;
            int col_off = 0;
            for (const auto& cp : comps) {
                block_col_offset[cp] = col_off;
                int d = 1;
                for (int part : cp) d *= static_cast<int>(A[part].basis_vectors.size());
                col_off += d;
            }

            std::vector<std::vector<Int>> D(Cdim[k - 1], std::vector<Int>(Cdim[k], 0));

            for (const auto& c : comps) {
                int kparts = static_cast<int>(c.size());

                std::vector<int> dims(kparts);
                for (int i = 0; i < kparts; ++i) dims[i] = static_cast<int>(A[c[i]].basis_vectors.size());

                int block_cols = 1;
                for (int d : dims) block_cols *= d;

                std::vector<int> stride(kparts, 1);
                for (int i = kparts - 2; i >= 0; --i) stride[i] = stride[i + 1] * dims[i + 1];

                for (int col_local = 0; col_local < block_cols; ++col_local) {
                    std::vector<int> idx(kparts, 0);
                    int rem = col_local;
                    for (int i = 0; i < kparts; ++i) {
                        idx[i] = rem / stride[i];
                        rem %= stride[i];
                    }

                    int global_col = block_col_offset[c] + col_local;

                    for (int i = 0; i + 1 < kparts; ++i) {
                        int a_deg = c[i], b_deg = c[i + 1];
                        const auto& va = A[a_deg].basis_vectors[idx[i]];
                        const auto& vb = A[b_deg].basis_vectors[idx[i + 1]];
                        auto prod_vec = qshuffle_product_vec(va, vb, a_deg, b_deg);
                        auto prod_coord = coordinates_in_basis(A[a_deg + b_deg], prod_vec);

                        std::vector<int> c2;
                        c2.reserve(kparts - 1);
                        for (int t = 0; t < i; ++t) c2.push_back(c[t]);
                        c2.push_back(c[i] + c[i + 1]);
                        for (int t = i + 2; t < kparts; ++t) c2.push_back(c[t]);

                        std::vector<int> idx2;
                        idx2.reserve(kparts - 1);
                        for (int t = 0; t < i; ++t) idx2.push_back(idx[t]);
                        idx2.push_back(0);
                        for (int t = i + 2; t < kparts; ++t) idx2.push_back(idx[t]);

                        std::vector<int> dims2(kparts - 1);
                        for (int t = 0; t < kparts - 1; ++t) dims2[t] = static_cast<int>(A[c2[t]].basis_vectors.size());
                        std::vector<int> stride2(kparts - 1, 1);
                        for (int t = kparts - 3; t >= 0; --t) stride2[t] = stride2[t + 1] * dims2[t + 1];

                        int eps = 0;
                        for (int t = 0; t <= i; ++t) eps += (c[t] - 1);
                        Int sign = (eps % 2 == 0) ? 1 : mod_norm(-1);
                        for (int b = 0; b < static_cast<int>(prod_coord.size()); ++b) {
                            if (prod_coord[b] == 0) continue;
                            idx2[i] = b;
                            int row_local = 0;
                            for (int t = 0; t < kparts - 1; ++t) row_local += idx2[t] * stride2[t];
                            int global_row = block_row_offset[c2] + row_local;
                            D[global_row][global_col] = mod_norm(D[global_row][global_col] + sign * prod_coord[b]);
                        }
                    }
                }
            }

            auto rt = rank_rectangular_wiedemann(D, threads, repeats, rng());
            rank_d[k] = rt.rank;
            timings[k] = rt;
            total_ms += rt.ms;
        }

        for (int k = 3; k <= n; ++k) {
            int max_allowed = Cdim[k - 1] - rank_d[k - 1];
            if (rank_d[k] > max_allowed) rank_d[k] = max_allowed;
        }

        for (int k = n; k >= 2; --k) {
            const auto& rt = timings[k];
            std::cout << "  d_" << k << ": B_" << k << " -> B_" << (k - 1)
                      << " (domain " << Cdim[k] << ", codomain " << Cdim[k - 1] << ")"
                      << ", matrix size " << rt.rows << "x" << rt.cols
                      << ", rank=" << rank_d[k]
                      << ", time=" << rt.ms << " ms\n";
        }

        std::cout << "  Cohomology dimensions H_k (k=1.." << n << "):\n";
        for (int k = 1; k <= n; ++k) {
            int rank_in = (k == n) ? 0 : rank_d[k + 1];
            int rank_out = (k == 1) ? 0 : rank_d[k];
            int h = (Cdim[k] - rank_out) - rank_in;
            std::cout << "    dim H_" << k << " = " << h << "\n";
        }
    }

    std::cout << "\nTotal Wiedemann time over all differentials: " << total_ms << " ms\n";
    return 0;
}
