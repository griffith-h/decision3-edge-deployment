/* Own CPU kernel implementing the public Transformers Qwen3.5 recurrent
 * gated-delta equations. No vendor binaries or model weights are changed.
 * Input FP32 layout [batch*heads, tokens, dim], output BF16 same layout.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline uint16x4_t bf16(float32x4_t v) {
    uint32x4_t u=vreinterpretq_u32_f32(v);
    uint32x4_t lsb=vandq_u32(vshrq_n_u32(u,16),vdupq_n_u32(1));
    uint32x4_t rounded=vshrq_n_u32(vaddq_u32(vaddq_u32(u,vdupq_n_u32(0x7fff)),lsb),16);
    uint32x4_t nan=vcgtq_u32(vandq_u32(u,vdupq_n_u32(0x7fffffff)),vdupq_n_u32(0x7f800000));
    rounded=vbslq_u32(nan,vdupq_n_u32(0x7fc0),rounded);
    return vmovn_u32(rounded);
}

int d3_scan(const float *restrict q,const float *restrict k,const float *restrict v,
            const float *restrict decay,const float *restrict beta,uint16_t *restrict out,
            int heads,int length,int kd,int vd,int tile,int threads,int fused) {
    if (heads<=0 || length<=0 || kd<=0 || kd>128 || vd<=0 || vd>128 || vd%4 ||
        (tile!=16 && tile!=32 && tile!=64 && tile!=128) || threads<1 || threads>3) return -1;
    int tiles=(vd+tile-1)/tile;
    /* Each work item owns its state and output columns. No persistent state,
     * cross-request cache, shared scratch, or Python tensor ownership here. */
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int job=0;job<heads*tiles;++job) {
        int h=job/tiles,first=(job%tiles)*tile,width=vd-first;
        if (width>tile) width=tile;
        float state[128*128] __attribute__((aligned(64)));
        memset(state,0,(size_t)kd*width*sizeof(float));
        if (fused>=4) {
            for (int t=0;t<length;++t) {
                size_t row=(size_t)h*length+t;
                const float *qt=q+row*kd,*kt=k+row*kd,*vt=v+row*vd+first;
                float d=decay[row],b=beta[row];
                int j=0;
                /* Four independent vector accumulators consume one complete
                 * cache line per row and avoid scalar reduction dependencies. */
                #define MAD(a,x,y) ((fused==5)?vfmaq_n_f32((a),(x),(y)):vaddq_f32((a),vmulq_n_f32((x),(y))))
                for (;j+16<=width;j+=16) {
                    float32x4_t p0=vdupq_n_f32(0),p1=p0,p2=p0,p3=p0;
                    for (int i=0;i<kd;++i) {
                        float *s=state+i*width+j;
                        float32x4_t s0=vmulq_n_f32(vld1q_f32(s),d),s1=vmulq_n_f32(vld1q_f32(s+4),d);
                        float32x4_t s2=vmulq_n_f32(vld1q_f32(s+8),d),s3=vmulq_n_f32(vld1q_f32(s+12),d);
                        vst1q_f32(s,s0);vst1q_f32(s+4,s1);vst1q_f32(s+8,s2);vst1q_f32(s+12,s3);
                        p0=MAD(p0,s0,kt[i]);p1=MAD(p1,s1,kt[i]);p2=MAD(p2,s2,kt[i]);p3=MAD(p3,s3,kt[i]);
                    }
                    float32x4_t d0=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j),p0),b);
                    float32x4_t d1=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j+4),p1),b);
                    float32x4_t d2=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j+8),p2),b);
                    float32x4_t d3=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j+12),p3),b);
                    float32x4_t o0=vdupq_n_f32(0),o1=o0,o2=o0,o3=o0;
                    for (int i=0;i<kd;++i) {
                        float *s=state+i*width+j;
                        float32x4_t s0=MAD(vld1q_f32(s),d0,kt[i]),s1=MAD(vld1q_f32(s+4),d1,kt[i]);
                        float32x4_t s2=MAD(vld1q_f32(s+8),d2,kt[i]),s3=MAD(vld1q_f32(s+12),d3,kt[i]);
                        vst1q_f32(s,s0);vst1q_f32(s+4,s1);vst1q_f32(s+8,s2);vst1q_f32(s+12,s3);
                        o0=MAD(o0,s0,qt[i]);o1=MAD(o1,s1,qt[i]);o2=MAD(o2,s2,qt[i]);o3=MAD(o3,s3,qt[i]);
                    }
                    uint16_t *dst=out+row*vd+first+j;
                    vst1_u16(dst,bf16(o0));vst1_u16(dst+4,bf16(o1));vst1_u16(dst+8,bf16(o2));vst1_u16(dst+12,bf16(o3));
                }
                for (;j<width;j+=4) {
                    float32x4_t p=vdupq_n_f32(0);
                    for (int i=0;i<kd;++i) {
                        float *s=state+i*width+j;
                        float32x4_t ss=vmulq_n_f32(vld1q_f32(s),d);vst1q_f32(s,ss);p=MAD(p,ss,kt[i]);
                    }
                    float32x4_t delta=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j),p),b),o=vdupq_n_f32(0);
                    for (int i=0;i<kd;++i) {
                        float *s=state+i*width+j;
                        float32x4_t ss=MAD(vld1q_f32(s),delta,kt[i]);vst1q_f32(s,ss);o=MAD(o,ss,qt[i]);
                    }
                    vst1_u16(out+row*vd+first+j,bf16(o));
                }
                #undef MAD
            }
            continue;
        }
        if (fused>=2) {
            /* Keep one output-vector reduction in NEON registers. Unlike the
             * row-oriented variant below, prediction/output are not reloaded
             * for every key row. Tile sizes keep the state hot in L1. */
            for (int t=0;t<length;++t) {
                size_t row=(size_t)h*length+t;
                const float *qt=q+row*kd,*kt=k+row*kd,*vt=v+row*vd+first;
                float d=decay[row],b=beta[row];
                for (int j=0;j<width;j+=4) {
                    float32x4_t p=vdupq_n_f32(0);
                    for (int i=0;i<kd;++i) {
                        float *s=state+i*width+j;
                        float32x4_t ss=vmulq_n_f32(vld1q_f32(s),d);
                        vst1q_f32(s,ss);
                        p=(fused==3)?vfmaq_n_f32(p,ss,kt[i]):vaddq_f32(p,vmulq_n_f32(ss,kt[i]));
                    }
                    float32x4_t delta=vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j),p),b);
                    float32x4_t o=vdupq_n_f32(0);
                    for (int i=0;i<kd;++i) {
                        float *s=state+i*width+j;
                        float32x4_t ss=vld1q_f32(s);
                        ss=(fused==3)?vfmaq_n_f32(ss,delta,kt[i]):vaddq_f32(ss,vmulq_n_f32(delta,kt[i]));
                        vst1q_f32(s,ss);
                        o=(fused==3)?vfmaq_n_f32(o,ss,qt[i]):vaddq_f32(o,vmulq_n_f32(ss,qt[i]));
                    }
                    vst1_u16(out+row*vd+first+j,bf16(o));
                }
            }
            continue;
        }
        for (int t=0;t<length;++t) {
            size_t row=(size_t)h*length+t;
            const float *qt=q+row*kd,*kt=k+row*kd,*vt=v+row*vd+first;
            float d=decay[row],b=beta[row];
            float prediction[128] __attribute__((aligned(64)))={0};
            float delta[128] __attribute__((aligned(64)));
            float output[128] __attribute__((aligned(64)))={0};
            for (int i=0;i<kd;++i) {
                float *s=state+i*width;
                for (int j=0;j<width;j+=4) {
                    float32x4_t ss=vmulq_n_f32(vld1q_f32(s+j),d);
                    vst1q_f32(s+j,ss);
                    float32x4_t p=vld1q_f32(prediction+j);
                    p=fused?vfmaq_n_f32(p,ss,kt[i]):vaddq_f32(p,vmulq_n_f32(ss,kt[i]));
                    vst1q_f32(prediction+j,p);
                }
            }
            for (int j=0;j<width;j+=4)
                vst1q_f32(delta+j,vmulq_n_f32(vsubq_f32(vld1q_f32(vt+j),vld1q_f32(prediction+j)),b));
            for (int i=0;i<kd;++i) {
                float *s=state+i*width;
                for (int j=0;j<width;j+=4) {
                    float32x4_t ss=vld1q_f32(s+j),dd=vld1q_f32(delta+j);
                    ss=fused?vfmaq_n_f32(ss,dd,kt[i]):vaddq_f32(ss,vmulq_n_f32(dd,kt[i]));
                    vst1q_f32(s+j,ss);
                    float32x4_t o=vld1q_f32(output+j);
                    o=fused?vfmaq_n_f32(o,ss,qt[i]):vaddq_f32(o,vmulq_n_f32(ss,qt[i]));
                    vst1q_f32(output+j,o);
                }
            }
            for (int j=0;j<width;j+=4)
                vst1_u16(out+row*vd+first+j,bf16(vld1q_f32(output+j)));
        }
    }
    return 0;
}
