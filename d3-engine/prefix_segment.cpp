// One compact KV head shared by every row and grouped query head. RKNN computes
// the two products; the running softmax spans all key blocks in FP32.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <vector>

// This frozen RK3576 bridge also provides its range-checked BF16 packer and NEON exp4.
#include "dynamic_matmul.cpp"

extern "C" int d3_segment(void *qk, void *pv, const uint16_t *q, const uint16_t *k,
                         const uint16_t *v, float *out, float *lse, int queries, int keys,
                         int dim, int block, float scale) {
    constexpr int tile = 128;
    if (!qk || !pv || !q || !k || !v || !out || !lse || queries < 1 || keys < 1 ||
        dim != 256 || block < 128 || block > 8192 || block % 128 || !std::isfinite(scale)) return -1;
    try {
        std::vector<__fp16> qt(tile * dim), key(dim * block), value(block * dim), prob(tile * block);
        std::vector<float> scores(tile * block), product(tile * dim), sum(queries, 0.f);
        std::fill(out, out + size_t(queries) * dim, 0.f);
        std::fill(lse, lse + queries, -INFINITY);
        for (int base = 0; base < keys; base += block) {
            int used = std::min(keys - base, block);
            std::fill(key.begin(), key.end(), __fp16(0));
            std::fill(value.begin(), value.end(), __fp16(0));
            for (int t = 0; t < used; ++t) {
                __fp16 row[256];
                size_t offset = size_t(base + t) * dim;
                if (!pack_row(k + offset, row, dim) || !pack_row(v + offset, &value[t * dim], dim)) return -2;
                for (int d = 0; d < dim; ++d) key[d * block + t] = row[d];
            }
            if (dm_set_b(qk, key.data()) || dm_set_b(pv, value.data())) return -1;
            for (int pos = 0; pos < queries; pos += tile) {
                int rows = std::min(tile, queries - pos);
                if (!pack_row(q + size_t(pos) * dim, qt.data(), rows * dim)) return -2;
                if (dm_run(qk, qt.data(), scores.data(), rows)) return -1;
                const int workers = std::max(1, omp_get_max_threads() / 2);
                #pragma omp parallel for num_threads(workers) schedule(dynamic, 8)
                for (int i = 0; i < rows; ++i) {
                    float *row = scores.data() + i * block;
                    float32x4_t max4 = vdupq_n_f32(lse[pos + i]);
                    int j = 0;
                    for (; j + 4 <= used; j += 4)
                        max4 = vmaxq_f32(max4, vmulq_n_f32(vld1q_f32(row + j), scale));
                    float maximum = vmaxvq_f32(max4);
                    for (; j < used; ++j) maximum = std::max(maximum, row[j] * scale);
                    float keep = std::exp(lse[pos + i] - maximum);
                    float32x4_t sum4 = vdupq_n_f32(0.f);
                    for (j = 0; j + 4 <= used; j += 4) {
                        float32x4_t e = exp4(vsubq_f32(vmulq_n_f32(vld1q_f32(row + j), scale), vdupq_n_f32(maximum)));
                        vst1_f16(prob.data() + i * block + j, vcvt_f16_f32(e));
                        sum4 = vaddq_f32(sum4, e);
                    }
                    float local = vaddvq_f32(sum4);
                    for (; j < used; ++j) {
                        float weight = std::exp(row[j] * scale - maximum);
                        local += weight;
                        prob[i * block + j] = static_cast<__fp16>(weight);
                    }
                    std::fill_n(prob.data() + i * block + used, block - used, __fp16(0));
                    sum[pos + i] = sum[pos + i] * keep + local;
                    lse[pos + i] = maximum;
                    float *dest = out + size_t(pos + i) * dim;
                    for (int d = 0; d < dim; ++d) dest[d] *= keep;
                }
                if (dm_run(pv, prob.data(), product.data(), rows)) return -1;
                for (int i = 0; i < rows * dim; ++i) out[size_t(pos) * dim + i] += product[i];
            }
        }
        for (int i = 0; i < queries; ++i) {
            if (!(sum[i] > 0.f) || !std::isfinite(sum[i])) return -2;
            float inverse = 1.f / sum[i];
            lse[i] += std::log(sum[i]);
            float *row = out + size_t(i) * dim;
            for (int d = 0; d < dim; ++d) row[d] *= inverse;
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
