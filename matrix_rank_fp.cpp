#include <iostream>
#include <vector>
#include <stdexcept>

using std::cin;
using std::cout;
using std::size_t;
using std::vector;
using int64 = long long;

static int64 normalize_mod(int64 x, int64 p) {
    x %= p;
    if (x < 0) {
        x += p;
    }
    return x;
}

static int64 mod_pow(int64 base, int64 exp, int64 mod) {
    int64 result = 1;
    base = normalize_mod(base, mod);

    while (exp > 0) {
        if (exp & 1LL) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1LL;
    }
    return result;
}

static int64 mod_inverse(int64 a, int64 p) {
    a = normalize_mod(a, p);
    if (a == 0) {
        throw std::invalid_argument("Zero has no multiplicative inverse modulo p");
    }

    // Fermat's little theorem (requires p to be prime).
    return mod_pow(a, p - 2, p);
}

static int rank_mod_p(vector<vector<int64>> matrix, int64 p) {
    const int n = static_cast<int>(matrix.size());
    if (n == 0) {
        return 0;
    }
    const int m = static_cast<int>(matrix[0].size());

    int rank = 0;
    int row = 0;

    for (int col = 0; col < m && row < n; ++col) {
        int pivot = row;
        while (pivot < n && normalize_mod(matrix[pivot][col], p) == 0) {
            ++pivot;
        }

        if (pivot == n) {
            continue;
        }

        std::swap(matrix[row], matrix[pivot]);

        const int64 pivot_value = normalize_mod(matrix[row][col], p);
        const int64 inv_pivot = mod_inverse(pivot_value, p);

        for (int j = col; j < m; ++j) {
            matrix[row][j] = normalize_mod(matrix[row][j] * inv_pivot, p);
        }

        for (int i = 0; i < n; ++i) {
            if (i == row) {
                continue;
            }
            const int64 factor = normalize_mod(matrix[i][col], p);
            if (factor == 0) {
                continue;
            }

            for (int j = col; j < m; ++j) {
                matrix[i][j] = normalize_mod(matrix[i][j] - factor * matrix[row][j], p);
            }
        }

        ++rank;
        ++row;
    }

    return rank;
}

#ifndef MATRIX_RANK_FP_NO_MAIN
int main() {
    std::ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    int64 p;

    // Input format:
    // n m p
    // followed by n rows, each with m integers.
    cin >> n >> m >> p;

    if (!cin || n < 0 || m < 0 || p <= 1) {
        cout << "Invalid input\n";
        return 1;
    }

    vector<vector<int64>> matrix(n, vector<int64>(m));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> matrix[i][j];
            if (!cin) {
                cout << "Invalid matrix input\n";
                return 1;
            }
            matrix[i][j] = normalize_mod(matrix[i][j], p);
        }
    }

    try {
        cout << rank_mod_p(matrix, p) << '\n';
    } catch (const std::exception& ex) {
        cout << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
#endif
