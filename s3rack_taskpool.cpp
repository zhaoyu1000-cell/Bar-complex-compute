// s3rack_taskpool.cpp
// Task pool over (k, monodromy-block g) for S3 transpositions rack.
// Computes rank(d_k) by summing block ranks, then outputs H^j.

// Build: g++ -O3 -std=c++20 -march=native -pipe -pthread s3rack_taskpool.cpp -o s3rack_taskpool
// Run:   ./s3rack_taskpool --n 6 --root-order 1 --sign 1 --prime 1000003 --threads 8
//        ./s3rack_taskpool --n 6 --root-order 1 --sign -1 --prime 1000003 --threads 8
//        ./s3rack_taskpool --n 6 --root-order 12 --root-exp 1 --sign 1 --threads 8

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sparse_rank_wiedemann.cpp"

using std::int64_t;

// -------------------- modular arithmetic --------------------
static inline int64_t mod_pow(int64_t a, int64_t e, int64_t p) {
    int64_t r = 1 % p;
    a %= p;
    while (e > 0) {
        if (e & 1) r = (int64_t)((__int128)r * a % p);
        a = (int64_t)((__int128)a * a % p);
        e >>= 1;
    }
    return r;
}
static inline int64_t mod_inv(int64_t a, int64_t p) {
    a %= p;
    if (a < 0) a += p;
    return mod_pow(a, p - 2, p);
}
static bool is_prime_ll(int64_t n) {
    if (n < 2) return false;
    if ((n % 2) == 0) return n == 2;
    for (int64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return false;
    return true;
}
static std::vector<int64_t> factor_distinct(int64_t n) {
    std::vector<int64_t> fac;
    for (int64_t d = 2; d * d <= n; d += (d == 2 ? 1 : 2)) {
        if (n % d == 0) {
            fac.push_back(d);
            while (n % d == 0) n /= d;
        }
    }
    if (n > 1) fac.push_back(n);
    return fac;
}
static int64_t primitive_root_mod_prime(int64_t p) {
    int64_t phi = p - 1;
    auto fac = factor_distinct(phi);
    for (int64_t g = 2; g < p; g++) {
        bool ok = true;
        for (auto q : fac) if (mod_pow(g, phi / q, p) == 1) { ok = false; break; }
        if (ok) return g;
    }
    throw std::runtime_error("No primitive root found");
}
static int64_t pick_prime_1_mod_m(int64_t m, int64_t start = 1000000) {
    if (m == 1) return 1000003;
    int64_t t = (start - 1) / m + 1;
    while (true) {
        int64_t cand = m * t + 1;
        if (is_prime_ll(cand)) return cand;
        t++;
    }
}
static int64_t q_scalar_in_gfp(int64_t p, int64_t m, int64_t exp, int sign) {
    if (sign != +1 && sign != -1) throw std::runtime_error("sign must be ±1");
    if (m == 1) return (sign == +1) ? 1 : (p - 1);
    if (p % m != 1) throw std::runtime_error("Need p ≡ 1 (mod m)");
    int64_t g = primitive_root_mod_prime(p);
    int64_t zeta = mod_pow(g, (p - 1) / m, p);
    int64_t q = mod_pow(zeta, ((exp % m) + m) % m, p);
    if (sign == -1) q = (p - q) % p;
    return q;
}

// -------------------- S3 / rack data --------------------
using Perm = std::array<int,3>;
static Perm compose_perm(const Perm& a, const Perm& b) {
    Perm r{};
    for (int i = 0; i < 3; i++) r[i] = a[b[i] - 1];
    return r;
}
static Perm inv_perm(const Perm& a) {
    Perm r{};
    for (int i = 0; i < 3; i++) r[a[i] - 1] = i + 1;
    return r;
}

// -------------------- shuffles and reduced swaps --------------------
static std::vector<int> reduced_swaps_for_perm(const std::vector<int>& w) {
    int m = (int)w.size();
    std::vector<int> arr(m);
    std::iota(arr.begin(), arr.end(), 0);
    std::vector<int> swaps;
    swaps.reserve(m*m);
    for (int i = 0; i < m; i++) {
        if (arr[i] == w[i]) continue;
        int j = (int)(std::find(arr.begin(), arr.end(), w[i]) - arr.begin());
        for (int k = j - 1; k >= i; k--) {
            std::swap(arr[k], arr[k+1]);
            swaps.push_back(k);
        }
    }
    return swaps;
}
static std::vector<std::pair<std::vector<int>, std::vector<int>>> shuffle_swaps(int p, int q) {
    int m = p + q;
    std::vector<std::pair<std::vector<int>, std::vector<int>>> res;
    std::vector<int> bit(m, 0);
    std::fill(bit.begin(), bit.begin() + p, 1);
    std::sort(bit.begin(), bit.end(), std::greater<int>());

    do {
        std::vector<int> w; w.reserve(m);
        int iu = 0, iv = 0;
        for (int r = 0; r < m; r++) {
            if (bit[r]) w.push_back(iu++);
            else        w.push_back(p + iv++);
        }
        res.push_back({w, reduced_swaps_for_perm(w)});
    } while (std::prev_permutation(bit.begin(), bit.end()));

    return res;
}

// -------------------- base-3 helpers --------------------
static std::vector<int> pow3_list(int n) {
    std::vector<int> p(n+1, 1);
    for (int i = 1; i <= n; i++) p[i] = p[i-1] * 3;
    return p;
}
static inline void decode3(int wid, int L, uint8_t* out) {
    for (int i = 0; i < L; i++) { out[i] = (uint8_t)(wid % 3); wid /= 3; }
}
static inline int encode3(const uint8_t* d, int L, const std::vector<int>& pow3) {
    int s = 0;
    for (int i = 0; i < L; i++) s += (int)d[i] * pow3[i];
    return s;
}
static inline int replace_segment(int wid, int start, int L, int out_id, const std::vector<int>& pow3) {
    int low  = wid % pow3[start];
    int high = wid / pow3[start + L];
    return low + out_id * pow3[start] + high * pow3[start + L];
}

// -------------------- qshuffle table --------------------
struct QShufflePQ {
    int p_len=0, q_len=0, out_len=0;
    // indexed by idx = u*3^{q_len} + v
    std::vector<std::vector<std::pair<int,int64_t>>> data;
};

static inline void apply_swaps_rack(uint8_t* d, int L, const std::vector<int>& swaps,
                                    const uint8_t op[3][3]) {
    for (int idx : swaps) {
        uint8_t x = d[idx], y = d[idx+1];
        d[idx]     = op[x][y];
        d[idx+1]   = x;
    }
}

static std::vector<std::vector<QShufflePQ>> precompute_qshuffle(
    int n, int64_t p, int64_t q_scalar, const uint8_t op[3][3], const std::vector<int>& pow3
) {
    std::vector<int64_t> qpow(n*n + 1, 1);
    for (int i = 1; i < (int)qpow.size(); i++) qpow[i] = (int64_t)((__int128)qpow[i-1] * q_scalar % p);

    // digits cache
    std::vector<std::vector<std::vector<uint8_t>>> digits(n+1);
    for (int L = 1; L <= n; L++) {
        digits[L].assign(pow3[L], std::vector<uint8_t>(L));
        for (int wid = 0; wid < pow3[L]; wid++) decode3(wid, L, digits[L][wid].data());
    }

    std::vector<std::vector<QShufflePQ>> table(n+1, std::vector<QShufflePQ>(n+1));
    std::array<uint8_t, 32> base{}, work{};

    for (int p_len = 1; p_len <= n-1; p_len++) {
        for (int q_len = 1; q_len <= n - p_len; q_len++) {
            int out_len = p_len + q_len;
            int out_sz  = pow3[out_len];

            QShufflePQ pq;
            pq.p_len = p_len; pq.q_len = q_len; pq.out_len = out_len;
            pq.data.assign(pow3[p_len] * pow3[q_len], {});

            auto sh = shuffle_swaps(p_len, q_len);

            std::vector<int64_t> acc(out_sz, 0);
            std::vector<int> touched; touched.reserve(128);

            for (int u = 0; u < pow3[p_len]; u++) {
                for (int v = 0; v < pow3[q_len]; v++) {
                    auto &ud = digits[p_len][u];
                    auto &vd = digits[q_len][v];
                    for (int i = 0; i < p_len; i++) base[i] = ud[i];
                    for (int j = 0; j < q_len; j++) base[p_len + j] = vd[j];

                    touched.clear();
                    for (auto &ws : sh) {
                        const auto &swaps = ws.second;
                        for (int t = 0; t < out_len; t++) work[t] = base[t];
                        apply_swaps_rack(work.data(), out_len, swaps, op);
                        int out_id = encode3(work.data(), out_len, pow3);

                        int64_t coeff = qpow[(int)swaps.size()];
                        if (acc[out_id] == 0) touched.push_back(out_id);
                        int64_t nv = acc[out_id] + coeff;
                        nv %= p; if (nv < 0) nv += p;
                        acc[out_id] = nv;
                    }

                    int idx = u * pow3[q_len] + v;
                    auto &vec = pq.data[idx];
                    vec.clear();
                    vec.reserve(touched.size());
                    for (int out_id : touched) {
                        int64_t c = acc[out_id];
                        if (c) vec.push_back({out_id, c});
                        acc[out_id] = 0;
                    }
                }
            }
            table[p_len][q_len] = std::move(pq);
        }
    }
    return table;
}

// -------------------- compositions --------------------
static void comps_rec(int n, int k, std::vector<int>& cur, std::vector<std::vector<int>>& out) {
    if (k == 1) { cur.push_back(n); out.push_back(cur); cur.pop_back(); return; }
    for (int first = 1; first <= n - k + 1; first++) {
        cur.push_back(first);
        comps_rec(n - first, k - 1, cur, out);
        cur.pop_back();
    }
}
static std::vector<std::vector<int>> compositions(int n, int k) {
    std::vector<std::vector<int>> out;
    std::vector<int> cur;
    comps_rec(n, k, cur, out);
    return out;
}
static std::vector<int> cuts_for_comp(const std::vector<int>& comp) {
    std::vector<int> cuts; cuts.reserve(comp.size()+1);
    cuts.push_back(0);
    int s = 0;
    for (int mj : comp) { s += mj; cuts.push_back(s); }
    return cuts;
}

// -------------------- monodromy blocks --------------------
struct MonoBlocks {
    std::vector<std::vector<int>> words_by_g; // 6 blocks
    std::vector<int> idx_in_block;
};
static MonoBlocks monodromy_blocks(int n, const std::vector<int>& pow3,
                                   const int mul6[6][6], const int trans_gid[3]) {
    int N = pow3[n];
    MonoBlocks mb;
    mb.words_by_g.assign(6, {});
    mb.idx_in_block.assign(N, -1);

    int e_id = 0;
    for (int wid = 0; wid < N; wid++) {
        int g = e_id, x = wid;
        for (int i = 0; i < n; i++) {
            int d = x % 3; x /= 3;
            g = mul6[g][trans_gid[d]];
        }
        mb.idx_in_block[wid] = (int)mb.words_by_g[g].size();
        mb.words_by_g[g].push_back(wid);
    }
    return mb;
}

// -------------------- sparse elimination --------------------
using SV = std::vector<std::pair<int,int64_t>>;

struct Spec { int start, p_len, q_len, out_len, new_pos; int64_t sgn; };

struct VecHash {
    size_t operator()(const std::vector<int>& v) const noexcept {
        size_t h = 0;
        for (int x : v) {
            h ^= std::hash<int>{}(x + 0x9e3779b9 + (int)(h<<6) + (int)(h>>2));
        }
        return h;
    }
};

struct SparseRectMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<SV> col_vectors;  // sparse columns: (row, value)
};

// Compile d_k restricted to monodromy block g into a sparse rectangular matrix.
static SparseRectMatrix compile_block_differential_matrix(
    int n, int k, int g, int64_t p,
    const std::vector<std::vector<QShufflePQ>>& qsh,
    const std::vector<int>& pow3,
    const MonoBlocks& mb
) {
    const auto comps_k = compositions(n, k);
    const auto comps_t = compositions(n, k - 1);

    std::unordered_map<std::vector<int>, int, VecHash> pos_t;
    pos_t.reserve(comps_t.size() * 2);
    for (int i = 0; i < (int)comps_t.size(); i++) pos_t[comps_t[i]] = i;

    std::vector<std::vector<Spec>> specs(comps_k.size());
    for (int ci = 0; ci < (int)comps_k.size(); ci++) {
        const auto &comp = comps_k[ci];
        const auto cuts = cuts_for_comp(comp);
        std::vector<Spec> sp; sp.reserve(k-1);
        for (int i = 1; i <= k-1; i++) {
            int start = cuts[i-1];
            int p_len = comp[i-1];
            int q_len = comp[i];
            int out_len = p_len + q_len;

            std::vector<int> new_comp;
            new_comp.insert(new_comp.end(), comp.begin(), comp.begin() + (i-1));
            new_comp.push_back(out_len);
            new_comp.insert(new_comp.end(), comp.begin() + (i+1), comp.end());

            int new_pos = pos_t[new_comp];
            int64_t sgn = ((i-1) & 1) ? (p - 1) : 1;
            sp.push_back({start, p_len, q_len, out_len, new_pos, sgn});
        }
        specs[ci] = std::move(sp);
    }

    const auto &words = mb.words_by_g[g];
    const int B = (int)words.size();
    if (B == 0) return {};

    const int rows = (int)comps_t.size() * B;
    std::vector<int64_t> acc(rows, 0);
    std::vector<int> touched; touched.reserve(256);

    // stamp trick to avoid duplicates in touched without hashing
    std::vector<int> mark(rows, 0);
    int stamp = 1;

    auto idx_map = [&](int wid) -> int { return mb.idx_in_block[wid]; };

    SparseRectMatrix mat;
    mat.rows = rows;
    mat.cols = (int)comps_k.size() * B;
    mat.col_vectors.assign(mat.cols, {});

    for (int ci = 0; ci < (int)comps_k.size(); ci++) {
        const auto &sp = specs[ci];
        for (int col_local = 0; col_local < B; col_local++) {
            int wid = words[col_local];
            touched.clear();
            stamp++;
            if (stamp == 0) { std::fill(mark.begin(), mark.end(), 0); stamp = 1; }

            for (const auto &ms : sp) {
                int u = (wid / pow3[ms.start]) % pow3[ms.p_len];
                int v = (wid / pow3[ms.start + ms.p_len]) % pow3[ms.q_len];
                int idx = u * pow3[ms.q_len] + v;
                const auto &lst = qsh[ms.p_len][ms.q_len].data[idx];
                int row_base = ms.new_pos * B;

                for (const auto &ow : lst) {
                    int out_id = ow.first;
                    int64_t coeff = ow.second;
                    int64_t val = (int64_t)((__int128)ms.sgn * coeff % p);
                    if (!val) continue;
                    int new_wid = replace_segment(wid, ms.start, ms.out_len, out_id, pow3);
                    int row = row_base + idx_map(new_wid);

                    if (mark[row] != stamp) {
                        mark[row] = stamp;
                        touched.push_back(row);
                    }
                    int64_t nv = acc[row] + val; nv %= p; if (nv < 0) nv += p;
                    acc[row] = nv;
                }
            }

            std::sort(touched.begin(), touched.end());

            SV vec;
            vec.reserve(touched.size());
            for (int r : touched) {
                int64_t c = acc[r];
                if (c) vec.push_back({r, c});
                acc[r] = 0;
            }
            int col = ci * B + col_local;
            mat.col_vectors[col] = std::move(vec);
        }
    }

    return mat;
}

// Rank via Wiedemann on square padding of D (no augmentation doubling).
static int rank_rectangular_wiedemann(const SparseRectMatrix& d, int64_t p, int repeats,
                                      int /*threads*/, std::uint64_t seed) {
    if (d.rows == 0 || d.cols == 0) return 0;
    const int n = std::max(d.rows, d.cols);
    sparse_wiedemann::SparsePairMatrix sq(n);
    std::vector<int> row_nnz(n, 0);
    for (int c = 0; c < d.cols; ++c) {
        for (const auto& [r, _] : d.col_vectors[c]) row_nnz[r]++;
    }
    for (int r = 0; r < d.rows; ++r) sq[r].reserve(row_nnz[r]);

    for (int c = 0; c < d.cols; ++c) {
        for (const auto& [r, v] : d.col_vectors[c]) {
            sq[r].push_back({c, v});
        }
    }

    std::mt19937_64 rng(seed);
    return sparse_wiedemann::rank_probabilistic(sq, p, rng, repeats);
}

// -------------------- binomial / printing --------------------
static __int128 binom_i128(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    __int128 r = 1;
    for (int i = 1; i <= k; i++) {
        r = r * (n - (k - i));
        r /= i;
    }
    return r;
}
static void print_i128(__int128 x) {
    if (x == 0) { std::cout << "0"; return; }
    if (x < 0) { std::cout << "-"; x = -x; }
    std::string s;
    while (x > 0) { s.push_back(char('0' + (int)(x % 10))); x /= 10; }
    std::reverse(s.begin(), s.end());
    std::cout << s;
}

// -------------------- main: task pool over monodromy blocks --------------------
struct Task { int g; };

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    int n = 3;
    int threads = 7;
    int sign = -1;
    int repeats = 5;
    int64_t m = 1, exp = 1, p = 0;

    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto need = [&](const char* opt) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + opt);
            return std::string(argv[++i]);
        };
        if (s == "--n") n = std::stoi(need("--n"));
        else if (s == "--threads") threads = std::stoi(need("--threads"));
        else if (s == "--root-order") m = std::stoll(need("--root-order"));
        else if (s == "--root-exp") exp = std::stoll(need("--root-exp"));
        else if (s == "--sign") sign = std::stoi(need("--sign"));
        else if (s == "--prime") p = std::stoll(need("--prime"));
        else if (s == "--repeats") repeats = std::stoi(need("--repeats"));
        else throw std::runtime_error("Unknown argument: " + s);
    }

    if (threads < 1) threads = 1;
    if (p == 0) p = pick_prime_1_mod_m(m);
    const int64_t q_scalar = q_scalar_in_gfp(p, m, exp, sign);

    std::cout << "n=" << n << " p=" << p << " m=" << m << " exp=" << exp
              << " sign=" << sign << " q_scalar=" << q_scalar
              << " threads=" << threads << "\n";

    // Build S3 in lex order and multiplication table
    std::vector<Perm> perms;
    Perm base = {1,2,3};
    do { perms.push_back(base); } while (std::next_permutation(base.begin(), base.end()));

    auto keyperm = [&](const Perm& a) { return a[0]*100 + a[1]*10 + a[2]; };
    std::unordered_map<int,int> perm_id;
    perm_id.reserve(6);
    for (int i = 0; i < 6; i++) perm_id[keyperm(perms[i])] = i;

    int mul6[6][6];
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) {
        Perm c = compose_perm(perms[i], perms[j]);
        mul6[i][j] = perm_id[keyperm(c)];
    }

    // transpositions: (12),(13),(23)
    Perm T12 = {2,1,3}, T13 = {3,2,1}, T23 = {1,3,2};
    int trans_gid[3] = { perm_id[keyperm(T12)], perm_id[keyperm(T13)], perm_id[keyperm(T23)] };

    // rack op on digits 0..2: x⋅y = xyx^{-1}
    uint8_t op3[3][3];
    std::vector<Perm> transP = {T12, T13, T23};
    for (int x = 0; x < 3; x++) for (int y = 0; y < 3; y++) {
        Perm z = compose_perm(compose_perm(transP[x], transP[y]), inv_perm(transP[x]));
        int idx = 0;
        for (int t = 0; t < 3; t++) if (transP[t] == z) idx = t;
        op3[x][y] = (uint8_t)idx;
    }

    const auto pow3 = pow3_list(n);

    // Precompute qshuffle
    auto t0 = std::chrono::high_resolution_clock::now();
    auto qsh = precompute_qshuffle(n, p, q_scalar, op3, pow3);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Monodromy blocks
    MonoBlocks mb = monodromy_blocks(n, pow3, mul6, trans_gid);

    std::vector<int> rank_d(n + 2, 0);           // rank_d[1]=rank_d[n+1]=0
    std::vector<double> rank_time_k(n + 2, 0.0); // wall time per differential d_k

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int k = 2; k <= n; k++) {
        std::vector<Task> tasks;
        tasks.reserve(6);
        for (int g = 0; g < 6; g++) {
            if (!mb.words_by_g[g].empty()) tasks.push_back({g});
        }

        auto k_start = std::chrono::high_resolution_clock::now();
        int rank_sum_k = 0;
        for (const Task& t : tasks) {
            SparseRectMatrix d_block = compile_block_differential_matrix(n, k, t.g, p, qsh, pow3, mb);
            std::uint64_t seed = 1469598103934665603ULL
                ^ (std::uint64_t)n * 1099511628211ULL
                ^ (std::uint64_t)k * 1315423911ULL
                ^ (std::uint64_t)t.g * 2654435761ULL;
            int rk = rank_rectangular_wiedemann(d_block, p, repeats, threads, seed);
            rank_sum_k += rk;
        }
        auto k_end = std::chrono::high_resolution_clock::now();

        rank_d[k] = rank_sum_k;
        rank_time_k[k] = std::chrono::duration<double>(k_end - k_start).count();
        std::cout << "[progress] finished d_" << k
                  << " rank=" << rank_d[k]
                  << " time=" << rank_time_k[k] << "s\n";
    }
    auto t3 = std::chrono::high_resolution_clock::now();

    double tq = std::chrono::duration<double>(t1 - t0).count();
    double tr = std::chrono::duration<double>(t3 - t2).count();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "qshuffle_pre=" << tq << "s  ranks_time=" << tr << "s\n\n";

    for (int k = 2; k <= n; k++) {
        std::cout << "rank(d_" << k << ")=" << rank_d[k]
                  << "  time=" << rank_time_k[k] << "s\n";
    }

    // Output H^j for cochain C^0=B_n -> ... -> C^{n-1}=B_1
    std::cout << "\nH^j dims (j=0..n-1):\n";
    for (int j = 0; j < n; j++) {
        int kk = n - j; // C^j = B_{kk}
        __int128 dimBk = binom_i128(n - 1, kk - 1) * (__int128)pow3[n];
        __int128 Hj = dimBk - (__int128)rank_d[kk] - (__int128)rank_d[kk + 1];
        std::cout << "H^" << j << " = ";
        print_i128(Hj);
        std::cout << "\n";
    }

    return 0;
}
