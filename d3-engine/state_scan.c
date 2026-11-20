/* Blockwise FP32 Gated DeltaNet on the board's NumPy ILP64 OpenBLAS.
 * One independent job per (batch, head). The state is [key_dim, value_dim].
 * The optional output checkpoint is private to the caller. */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum { ROW_MAJOR = 101, NO_TRANS = 111, TRANS = 112, LOWER = 122,
       UNIT_DIAG = 132, LEFT = 141, BLOCK = 32, MAX_DIM = 128 };

extern void scipy_cblas_sgemm64_(int, int, int, int64_t, int64_t, int64_t, float,
                                 const float *, int64_t, const float *, int64_t,
                                 float, float *, int64_t);
extern void scipy_cblas_strsm64_(int, int, int, int, int, int64_t, int64_t, float,
                                 const float *, int64_t, float *, int64_t);

static uint16_t bfloat16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U)
        return (uint16_t)((bits >> 16) | 0x40U);
    return (uint16_t)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

static void scan_head(const float *q, const float *k, const float *v, const float *decay,
                      const float *beta, const float *initial, float *final, uint16_t *out,
                      int length, int kd, int vd) {
    float state[MAX_DIM * MAX_DIM], pred[BLOCK * MAX_DIM], read[BLOCK * MAX_DIM];
    float lower[BLOCK * BLOCK], attn[BLOCK * BLOCK], weighted_keys[BLOCK * MAX_DIM];
    size_t state_bytes = (size_t)kd * vd * sizeof(float);
    if (initial)
        memcpy(state, initial, state_bytes);
    else
        memset(state, 0, state_bytes);

    for (int pos = 0; pos < length; pos += BLOCK) {
        int n = length - pos < BLOCK ? length - pos : BLOCK;
        const float *keys = k + (size_t)pos * kd, *queries = q + (size_t)pos * kd;
        scipy_cblas_sgemm64_(ROW_MAJOR, NO_TRANS, TRANS, n, n, kd, 1, keys, kd, keys, kd, 0, lower, n);
        scipy_cblas_sgemm64_(ROW_MAJOR, NO_TRANS, TRANS, n, n, kd, 1, queries, kd, keys, kd, 0, attn, n);
        scipy_cblas_sgemm64_(ROW_MAJOR, NO_TRANS, NO_TRANS, n, vd, kd, 1, keys, kd, state, vd, 0, pred, vd);
        scipy_cblas_sgemm64_(ROW_MAJOR, NO_TRANS, NO_TRANS, n, vd, kd, 1, queries, kd, state, vd, 0, read, vd);
        float prefix_decay = 1;
        for (int i = 0; i < n; ++i) {
            prefix_decay *= decay[pos + i];
            for (int j = 0; j < vd; ++j) {
                size_t t = (size_t)i * vd + j;
                pred[t] = beta[pos + i] * (v[(size_t)pos * vd + t] - prefix_decay * pred[t]);
                read[t] *= prefix_decay;
            }
            float between = 1;
            for (int j = i; j >= 0; --j) {
                lower[i*n+j] *= between * beta[pos + i];
                attn[i*n+j] *= between;
                between *= decay[pos + j];
            }
            memset(attn + i*n + i + 1, 0, (size_t)(n-i-1) * sizeof(float));
        }
        /* Solve for the value corrections in place, then read this block. */
        scipy_cblas_strsm64_(ROW_MAJOR, LEFT, LOWER, NO_TRANS, UNIT_DIAG, n, vd, 1, lower, n, pred, vd);
        scipy_cblas_sgemm64_(ROW_MAJOR, NO_TRANS, NO_TRANS, n, vd, n, 1, attn, n, pred, vd, 1, read, vd);
        for (int i = 0; i < n * vd; ++i)
            out[(size_t)pos * vd + i] = bfloat16(read[i]);

        if (pos + n < length || final) {
            float tail = 1;
            for (int i = n - 1; i >= 0; --i) {
                for (int j = 0; j < kd; ++j)
                    weighted_keys[i*kd+j] = tail * keys[i*kd+j];
                tail *= decay[pos + i];
            }
            scipy_cblas_sgemm64_(ROW_MAJOR, TRANS, NO_TRANS, kd, vd, n, 1, weighted_keys, kd,
                                 pred, vd, prefix_decay, state, vd);
        }
    }
    if (final)
        memcpy(final, state, state_bytes);
}

int d3_scan_state(const float *q, const float *k, const float *v, const float *decay,
                  const float *beta, const float *initial, float *final, uint16_t *out,
                  int heads, int length, int kd, int vd) {
    if (!q || !k || !v || !decay || !beta || !out || heads <= 0 || length <= 0 ||
        kd <= 0 || kd > MAX_DIM || vd <= 0 || vd > MAX_DIM)
        return -1;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int h = 0; h < heads; ++h) {
        size_t row = (size_t)h * length, offset = (size_t)h * kd * vd;
        scan_head(q + row*kd, k + row*kd, v + row*vd, decay + row, beta + row,
                  initial ? initial + offset : NULL, final ? final + offset : NULL,
                  out + row*vd, length, kd, vd);
    }
    return 0;
}
