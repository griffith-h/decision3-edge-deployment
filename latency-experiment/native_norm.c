/* CPU-only experimental fusion. No vendor runtime code. */
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <arm_neon.h>

static inline float bf(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float out;
    memcpy(&out, &bits, 4);
    return out;
}
static inline uint16_t rb(float v) {
    uint32_t bits;
    memcpy(&bits, &v, 4);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return (bits >> 16) | 0x40;
    return (uint16_t)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
static inline float32x4_t bf4(const uint16_t *x) {
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(x), 16));
}

/* kind=0: zero-centered norm; kind=1: gated norm, including BF16 boundaries. */
void d3_norm(const uint16_t *x, const uint16_t *w, const uint16_t *gate,
             uint16_t *out, int rows, int dim, float eps, int kind) {
    #pragma omp parallel for num_threads(4) if(rows > 16)
    for (int r = 0; r < rows; ++r) {
        const uint16_t *in = x + (size_t)r * dim;
        uint16_t *dst = out + (size_t)r * dim;
        float32x4_t acc = vdupq_n_f32(0);
        int i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t v = bf4(in+i);
            acc = vaddq_f32(acc, vmulq_f32(v, v));
        }
        float sum = vaddvq_f32(acc);
        for (; i < dim; ++i) { float v = bf(in[i]); sum += v*v; }
        float scale = 1.0f / sqrtf(sum / dim + eps);
        for (i = 0; i < dim; ++i) {
            float value = bf(in[i]) * scale;
            if (kind == 0) {
                value *= 1.0f + bf(w[i]);
            } else {
                value = bf(rb(value));
                value = bf(rb(value * bf(w[i])));
                float g = bf(gate[(size_t)r * dim + i]);
                value *= g / (1.0f + expf(-g));
            }
            dst[i] = rb(value);
        }
    }
}

static inline uint16x4_t rb4(float32x4_t v) {
    uint32x4_t u = vreinterpretq_u32_f32(v);
    uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1));
    uint32x4_t rounded = vshrq_n_u32(vaddq_u32(vaddq_u32(u, vdupq_n_u32(0x7fff)), lsb), 16);
    uint32x4_t nan = vcgtq_u32(vandq_u32(u, vdupq_n_u32(0x7fffffff)), vdupq_n_u32(0x7f800000));
    rounded = vbslq_u32(nan, vorrq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(0x40)), rounded);
    return vmovn_u32(rounded);
}

/* Original 4-tap causal conv, FP32 accumulation then one BF16 boundary. */
void d3_conv4(const uint16_t *x, const uint16_t *w, const uint16_t *bias,
              uint16_t *out, int batch, int channels, int length) {
    #pragma omp parallel for num_threads(4) if(channels * batch > 16)
    for (int row = 0; row < batch * channels; ++row) {
        int c = row % channels;
        const uint16_t *in = x + (size_t)row * length;
        uint16_t *dst = out + (size_t)row * length;
        float weights[4];
        for (int i = 0; i < 4; ++i) weights[i] = bf(w[c*4+i]);
        int t = 0;
        for (; t < length && t < 3; ++t) {
            float v = bf(in[t]) * weights[3];
            for (int o = 1; o <= t; ++o) v += bf(in[t-o]) * weights[3-o];
            if (bias) v += bf(bias[c]);
            dst[t] = rb(v);
        }
        for (; t + 4 <= length; t += 4) {
            float32x4_t v = vmulq_n_f32(bf4(in+t), weights[3]);
            for (int o = 1; o < 4; ++o) v = vaddq_f32(v, vmulq_n_f32(bf4(in+t-o), weights[3-o]));
            if (bias) v = vaddq_f32(v, vdupq_n_f32(bf(bias[c])));
            vst1_u16(dst+t, rb4(v));
        }
        for (; t < length; ++t) {
            float v = bf(in[t]) * weights[3];
            for (int o = 1; o < 4; ++o) v += bf(in[t-o]) * weights[3-o];
            if (bias) v += bf(bias[c]);
            dst[t] = rb(v);
        }
    }
}
