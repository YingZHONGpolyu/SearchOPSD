// Exact matching-block count for difflib.SequenceMatcher(autojunk=False, isjunk=None).
// One CUDA block handles one pair; lanes scan different common-substring diagonals.
#include <stdint.h>

extern "C" __global__ void sequence_matches(
    const int32_t* chars_a, const int32_t* chars_b,
    const int64_t* offsets_a, const int64_t* offsets_b,
    int32_t* matches, int32_t* overflow, int count) {
    int pair = blockIdx.x;
    if (pair >= count) return;
    int lane = threadIdx.x;
    const int32_t* a = chars_a + offsets_a[pair];
    const int32_t* b = chars_b + offsets_b[pair];
    int na = offsets_a[pair + 1] - offsets_a[pair];
    int nb = offsets_b[pair + 1] - offsets_b[pair];
    // The stack contains disjoint rectangles produced by matching-block splits.
    __shared__ int stack[256][4];
    __shared__ int top, total, active, alo, ahi, blo, bhi;
    __shared__ int lengths[256], starts_a[256], starts_b[256];
    if (lane == 0) {
        top = 1; total = 0;
        stack[0][0] = 0; stack[0][1] = na;
        stack[0][2] = 0; stack[0][3] = nb;
        overflow[pair] = 0;
    }
    __syncthreads();
    while (true) {
        if (lane == 0) {
            active = top > 0;
            if (active) {
                --top;
                alo = stack[top][0]; ahi = stack[top][1];
                blo = stack[top][2]; bhi = stack[top][3];
            }
        }
        __syncthreads();
        if (!active) break;
        int best = 0, best_a = alo, best_b = blo;
        int an = ahi - alo, bn = bhi - blo;
        for (int diagonal = lane; diagonal < an + bn - 1; diagonal += blockDim.x) {
            int difference = diagonal - (bn - 1);
            int i = alo + (difference > 0 ? difference : 0);
            int j = blo + (difference < 0 ? -difference : 0);
            int run = 0;
            for (; i < ahi && j < bhi; ++i, ++j) {
                run = a[i] == b[j] ? run + 1 : 0;
                int start_a = i - run + 1, start_b = j - run + 1;
                // difflib resolves equal-length matches by earliest a, then earliest b.
                if (run > best || (run > 0 && run == best &&
                    (start_a < best_a || (start_a == best_a && start_b < best_b)))) {
                    best = run; best_a = start_a; best_b = start_b;
                }
            }
        }
        lengths[lane] = best; starts_a[lane] = best_a; starts_b[lane] = best_b;
        __syncthreads();
        if (lane == 0) {
            int size = 0, i = alo, j = blo;
            for (int t = 0; t < blockDim.x; ++t) {
                if (lengths[t] > size || (lengths[t] > 0 && lengths[t] == size &&
                    (starts_a[t] < i || (starts_a[t] == i && starts_b[t] < j)))) {
                    size = lengths[t]; i = starts_a[t]; j = starts_b[t];
                }
            }
            if (size > 0) {
                total += size;
                int needed = (alo < i && blo < j) + (i + size < ahi && j + size < bhi);
                if (top + needed > 256) {
                    overflow[pair] = 1; top = 0;
                } else {
                    if (alo < i && blo < j) {
                        stack[top][0] = alo; stack[top][1] = i;
                        stack[top][2] = blo; stack[top][3] = j; ++top;
                    }
                    if (i + size < ahi && j + size < bhi) {
                        stack[top][0] = i + size; stack[top][1] = ahi;
                        stack[top][2] = j + size; stack[top][3] = bhi; ++top;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (lane == 0) matches[pair] = total;
}
