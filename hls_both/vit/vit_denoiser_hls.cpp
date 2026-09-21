#include "vit_denoiser_hls.h"
#include <hls_math.h>

// ===========================================================================
// On-chip state.
//
// Two token buffers ping-pong across sublayers (attn writes B, FFN reads B
// writes A, and so on) so no sublayer aliases its own input. Each is
// N_TOK x VK_DIM x 2 B = 294 KB at 16-bit -- the dominant BRAM cost, and the
// reason weights must stream instead of being resident.
// ===========================================================================
static data_t tokA[N_TOK][VK_DIM];
static data_t tokB[N_TOK][VK_DIM];

// Streamed per-layer weights, double-buffered in load_layer_weights.
static data_t w_in[3*VK_DIM][VK_DIM],  b_in[3*VK_DIM];
static data_t w_out[VK_DIM][VK_DIM],   b_out[VK_DIM];
static data_t w_l1[D_FF][VK_DIM],  b_l1[D_FF];
static data_t w_l2[VK_DIM][D_FF];
static data_t b_l2[VK_DIM];
static data_t g_n1[VK_DIM], be_n1[VK_DIM], g_n2[VK_DIM], be_n2[VK_DIM];
// Hidden activations for one FFN tile, and the running Linear2 accumulator
// across tiles (Linear2 sums over all D_FF, so partial sums must persist).
static data_t hid_t[TILE_F];

// Q/K/V for one layer. 3 x 784 x 192 x 2 B = 882 KB -- too large to keep all
// three, so Q is recomputed per tile while K and V persist (they are reused by
// every query tile, Q is not).
static data_t Kbuf[N_TOK][VK_DIM], Vbuf[N_TOK][VK_DIM];

// ===========================================================================
// 2D sin-cos positional encoding, computed on the fly.
//
// PyTorch builds this with torch.sin/cos at fp32. hls::sinf is expensive
// (~90 LUT each) but this runs once per image over 784x192 values, so it is
// cheaper in area than a 294 KB ROM. Matches _pos_embed(pos_mode="2d").
// ===========================================================================
static data_t pos_embed(int tok, int c) {
#pragma HLS INLINE off
    const int d = VK_DIM / 2;                     // 96
    int gy = tok / GRID_W, gx = tok % GRID_W;
    int i  = (c < d) ? c : (c - d);            // index within the half
    int p  = (c < d) ? gy : gx;                // row half / column half
    float freq = hls::powf(10000.0f, (float)(2 * (i / 2)) / (float)d);
    float ang  = (float)p / freq;
    return (data_t)((i % 2 == 0) ? hls::sinf(ang) : hls::cosf(ang));
}

// ===========================================================================
// Stage 1: patch embed. Conv2d(3,192,k=4,s=4) is a non-overlapping patch
// gather, so it is a dense 48->192 matmul per patch, not a sliding window.
// Positional encoding is folded in here to avoid a second pass over tokA.
// ===========================================================================
static void patch_embed(const float *img_in, const float *w) {
    const float *ew = w;
    const float *eb = w + VK_DIM*C_IN*VK_PATCH*VK_PATCH;

    EMB_TOK: for (int t = 0; t < N_TOK; ++t) {
        int gy = t / GRID_W, gx = t % GRID_W;
        // Gather the 4x4x3 patch once; reused across all 192 output channels.
        data_t patch[C_IN*VK_PATCH*VK_PATCH];
#pragma HLS ARRAY_PARTITION variable=patch complete dim=1
        for (int c = 0; c < C_IN; ++c)
            for (int py = 0; py < VK_PATCH; ++py)
                for (int px = 0; px < VK_PATCH; ++px) {
#pragma HLS PIPELINE II=1
                    int iy = gy*VK_PATCH + py, ix = gx*VK_PATCH + px;
                    patch[(c*VK_PATCH + py)*VK_PATCH + px] =
                        (data_t)img_in[(c*IMG_H + iy)*IMG_W + ix];
                }

        EMB_OC: for (int oc = 0; oc < VK_DIM; ++oc) {
#pragma HLS PIPELINE II=1
            acc_t acc = (acc_t)eb[oc];
            for (int k = 0; k < C_IN*VK_PATCH*VK_PATCH; ++k) {
#pragma HLS UNROLL
                acc += (acc_t)patch[k] * (acc_t)ew[oc*C_IN*VK_PATCH*VK_PATCH + k];
            }
            tokA[t][oc] = (data_t)acc + pos_embed(t, oc);
        }
    }
}

// ===========================================================================
// Stream one layer's weights from DDR into BRAM.
// 1.24 MB per layer at 16-bit -- this is the bandwidth bottleneck, and the
// reason the design is memory-bound rather than DSP-bound.
// ===========================================================================
static void load_layer_weights(const float *w) {
    const float *p = w;
    LW_IN:  for (int i = 0; i < 3*VK_DIM; ++i)
                for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS PIPELINE II=1
                    w_in[i][j] = (data_t)p[i*VK_DIM + j]; }
    p += 3*VK_DIM*VK_DIM;
    LB_IN:  for (int i = 0; i < 3*VK_DIM; ++i) { 
#pragma HLS PIPELINE II=1
                b_in[i] = (data_t)p[i]; }
    p += 3*VK_DIM;
    LW_OUT: for (int i = 0; i < VK_DIM; ++i)
                for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS PIPELINE II=1
                    w_out[i][j] = (data_t)p[i*VK_DIM + j]; }
    p += VK_DIM*VK_DIM;
    LB_OUT: for (int i = 0; i < VK_DIM; ++i) {
#pragma HLS PIPELINE II=1
                b_out[i] = (data_t)p[i]; }
    p += VK_DIM;
    LN1:    for (int i = 0; i < VK_DIM; ++i) {
#pragma HLS PIPELINE II=1
                g_n1[i] = (data_t)p[i]; be_n1[i] = (data_t)p[VK_DIM + i]; }
    p += 2*VK_DIM;
    LW_L1:  for (int i = 0; i < D_FF; ++i)
                for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS PIPELINE II=1
                    w_l1[i][j] = (data_t)p[i*VK_DIM + j]; }
    p += D_FF*VK_DIM;
    LB_L1:  for (int i = 0; i < D_FF; ++i) {
#pragma HLS PIPELINE II=1
                b_l1[i] = (data_t)p[i]; }
    p += D_FF;
    LW_L2:  for (int i = 0; i < VK_DIM; ++i)
                for (int j = 0; j < D_FF; ++j) {
#pragma HLS PIPELINE II=1
                    w_l2[i][j] = (data_t)p[i*D_FF + j]; }
    p += VK_DIM*D_FF + VK_DIM;
    LN2:    for (int i = 0; i < VK_DIM; ++i) {
#pragma HLS PIPELINE II=1
                g_n2[i] = (data_t)p[i]; be_n2[i] = (data_t)p[VK_DIM + i]; }
}

// ===========================================================================
// K and V for every token. Computed once per layer, reused by all query tiles.
// w_in is [Q;K;V] stacked, matching nn.MultiheadAttention.in_proj_weight.
// ===========================================================================
// K/V for every head, computed ONCE per layer. The ZCU104 has 624 BRAM18
// blocks (2808 KB), so all 588 KB of K/V stays resident -- recomputing per
// head would cost +2.31 GFLOP (38% of total) to save memory we have.
static void compute_kv(const data_t src[N_TOK][VK_DIM]) {
    KV_TOK: for (int t = 0; t < N_TOK; ++t) {
        KV_OC: for (int oc = 0; oc < VK_DIM; ++oc) {
#pragma HLS PIPELINE II=1
            acc_t ak = (acc_t)b_in[VK_DIM + oc];
            acc_t av = (acc_t)b_in[2*VK_DIM + oc];
            for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS UNROLL factor=UNROLL_F
                data_t s = src[t][j];
                ak += (acc_t)s * (acc_t)w_in[VK_DIM + oc][j];
                av += (acc_t)s * (acc_t)w_in[2*VK_DIM + oc][j];
            }
            Kbuf[t][oc] = (data_t)ak;
            Vbuf[t][oc] = (data_t)av;
        }
    }
}

// ===========================================================================
// Multi-head self-attention over one tile of TILE_N query rows.
//
// Softmax uses the max-subtraction form. In fixed point this is not merely a
// numerical nicety: pre-softmax logits reach ~20, and expf(20) overflows
// ap_fixed<24,10>. Subtracting the row max bounds the argument at 0.
// ===========================================================================
// ===========================================================================
// Multi-head self-attention, head-outer / tile-inner.
//
// Loop order matters for BRAM: K/V are held for one head only (98 KB vs
// 588 KB), so the head loop must be OUTSIDE the query-tile loop or K/V would
// be recomputed once per tile (49x) instead of once per head (6x).
//
// Softmax uses the max-subtraction form. In fixed point this is load-bearing,
// not cosmetic: pre-softmax logits reach ~20 and expf(20) overflows
// ap_fixed<24,10>. Subtracting the row max bounds the argument at 0.
// ===========================================================================
static data_t Qb[TILE_N][VK_DIM];
static exp_t  scores[TILE_N][N_TOK];
// Per-row softmax statistics, carried between the score pass and the AV pass
// so neither needs its own sweep over N_TOK.
static exp_t  rowmax[TILE_N];
static float  rowinv[TILE_N];

static void attention_all(const data_t src[N_TOK][VK_DIM], data_t dst[N_TOK][VK_DIM]) {
    compute_kv(src);                  // once per layer, all heads resident

    const float inv_sqrt_d = 1.0f / hls::sqrtf((float)VK_DHEAD);

    // Tile-OUTER / head-INNER: a query tile finishes all six heads before the
    // next tile starts, so the context accumulator is ctx[TILE_N][VK_DIM] (12 KB)
    // instead of ctxall[N_TOK][VK_DIM] (588 KB). K/V are resident, so re-reading
    // them per tile costs BRAM bandwidth, not FLOPs.
    TILES: for (int t0 = 0; t0 < N_TOK; t0 += TILE_N) {
        int tn = (t0 + TILE_N <= N_TOK) ? TILE_N : (N_TOK - t0);

        acc_t ctx[TILE_N][VK_DIM];
#pragma HLS ARRAY_PARTITION variable=ctx cyclic factor=PF_TOK dim=2
        CZ: for (int u = 0; u < tn; ++u)
            for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
                ctx[u][c] = (acc_t)0;
            }

        HEAD: for (int h = 0; h < VK_HEADS; ++h) {
            int hoff = h * VK_DHEAD;

            Q_TOK: for (int u = 0; u < tn; ++u) {
                Q_OC: for (int c = 0; c < VK_DHEAD; ++c) {
#pragma HLS PIPELINE II=1
                    int oc = hoff + c;
                    acc_t a = (acc_t)b_in[oc];
                    for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS UNROLL factor=UNROLL_F
                        a += (acc_t)src[t0 + u][j] * (acc_t)w_in[oc][j];
                    }
                    Qb[u][oc] = (data_t)a;
                }
            }

            // The row max is produced here rather than in a second pass:
            // softmax was measured at 38% of total runtime, and its three
            // sequential 784-iteration passes were the largest non-MAC cost.
            SC_U: for (int u = 0; u < tn; ++u) {
                exp_t mx = (exp_t)-1e30f;
                SC_T: for (int t = 0; t < N_TOK; ++t) {
#pragma HLS PIPELINE II=1
                    acc_t a = (acc_t)0;
                    for (int c = 0; c < VK_DHEAD; ++c) {
#pragma HLS UNROLL factor=UNROLL_F
                        a += (acc_t)Qb[u][hoff + c] * (acc_t)Kbuf[t][hoff + c];
                    }
                    exp_t sv = (exp_t)((float)a * inv_sqrt_d);
                    scores[u][t] = sv;
                    if (sv > mx) mx = sv;
                }
                rowmax[u] = mx;
            }

            // max-subtraction softmax: load-bearing in fixed point, since
            // expf(20) overflows ap_fixed<24,10>.
            // exp() pass only. The max already came from SC_T, and the 1/sum
            // normalisation is folded into the AV accumulation below, so two
            // of the original three 784-iteration passes are gone.
            SM_U: for (int u = 0; u < tn; ++u) {
                exp_t mx = rowmax[u];
                exp_t sum = (exp_t)0;
                SM_EXP: for (int t = 0; t < N_TOK; ++t) {
#pragma HLS PIPELINE II=1
                    exp_t e = (exp_t)hls::expf((float)(scores[u][t] - mx));
                    scores[u][t] = e;
                    sum += e;
                }
                rowinv[u] = 1.0f / (float)sum;
            }

            AV_U: for (int u = 0; u < tn; ++u) {
                float rinv = rowinv[u];
                AV_T: for (int t = 0; t < N_TOK; ++t) {
#pragma HLS PIPELINE II=1
                    acc_t p = (acc_t)((float)scores[u][t] * rinv);
                    for (int c = 0; c < VK_DHEAD; ++c) {
#pragma HLS UNROLL factor=UNROLL_F
                        ctx[u][hoff + c] += p * (acc_t)Vbuf[t][hoff + c];
                    }
                }
            }
        }

        // ---- out-proj, residual, LayerNorm for this tile -----------------
        OP_U: for (int u = 0; u < tn; ++u) {
            int t = t0 + u;
            data_t y[VK_DIM];
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=PF_TOK dim=1
            OP_OC: for (int oc = 0; oc < VK_DIM; ++oc) {
#pragma HLS PIPELINE II=1
                acc_t a = (acc_t)b_out[oc];
                for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS UNROLL factor=UNROLL_F
                    a += ctx[u][j] * (acc_t)w_out[oc][j];
                }
                y[oc] = src[t][oc] + (data_t)a;
            }
            acc_t mean = (acc_t)0;
            LN_M: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
                mean += (acc_t)y[c];
            }
            float mu = (float)mean / (float)VK_DIM;
            acc_t var = (acc_t)0;
            LN_V: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
                float d = (float)y[c] - mu;
                var += (acc_t)(d*d);
            }
            float inv_std = 1.0f / hls::sqrtf((float)var / (float)VK_DIM + 1e-5f);
            LN_A: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
                dst[t][c] = (data_t)((((float)y[c] - mu) * inv_std)
                                     * (float)g_n1[c] + (float)be_n1[c]);
            }
        }
    }
}

// ===========================================================================
// Feed-forward block, streamed in TILE_F hidden-row tiles.
//
// w_l1 (768x192) + w_l2 (192x768) is 576 KB per layer at 16-bit, which cannot
// sit alongside the 588 KB token buffers on a 1375 KB device. Instead each
// tile of TILE_F hidden rows is fetched from DDR, used, and discarded: 96 KB
// resident. Linear2 sums over all D_FF, so its partial sums accumulate in
// ff_acc across tiles and the residual+LayerNorm runs only after the last one.
// ===========================================================================
static void load_ffn_tile(const float *lw, int f0) {
    const float *p1 = lw + OFF_LIN1_W + (size_t)f0*VK_DIM;
    const float *pb = lw + OFF_LIN1_B + f0;
    const float *p2 = lw + OFF_LIN2_W;
    T_W1: for (int i = 0; i < TILE_F; ++i)
              for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS PIPELINE II=1
                  w_l1[i][j] = (data_t)p1[(size_t)i*VK_DIM + j]; }
    T_B1: for (int i = 0; i < TILE_F; ++i) {
#pragma HLS PIPELINE II=1
              b_l1[i] = (data_t)pb[i]; }
    // w_l2 is [VK_DIM][D_FF]; this tile needs columns f0 .. f0+TILE_F-1
    T_W2: for (int i = 0; i < VK_DIM; ++i)
              for (int j = 0; j < TILE_F; ++j) {
#pragma HLS PIPELINE II=1
                  w_l2[i][j] = (data_t)p2[(size_t)i*D_FF + f0 + j]; }
}

static void ffn_block(const data_t src[N_TOK][VK_DIM], data_t dst[N_TOK][VK_DIM],
                      const float *lw) {
    const float *p_b2 = lw + OFF_LIN2_B;

    // Token-OUTER: each token completes all its FFN tiles before the next
    // starts, so the Linear2 partial sum lives in a VK_DIM-wide local register
    // instead of an N_TOK x VK_DIM buffer. Saves 588 KB (131 BRAM18 blocks).
    FF_TOK: for (int t = 0; t < N_TOK; ++t) {
        acc_t acc2[VK_DIM];
#pragma HLS ARRAY_PARTITION variable=acc2 cyclic factor=PF_TOK dim=1
        FA_INIT: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
            acc2[c] = (acc_t)p_b2[c];
        }

        FTILE: for (int f0 = 0; f0 < D_FF; f0 += TILE_F) {
            // weights for this tile are already resident (loaded per layer)
            FF_L1: for (int u = 0; u < TILE_F; ++u) {
#pragma HLS PIPELINE II=1
                acc_t a = (acc_t)b_l1[f0 + u];
                for (int j = 0; j < VK_DIM; ++j) {
#pragma HLS UNROLL factor=UNROLL_F
                    a += (acc_t)src[t][j] * (acc_t)w_l1[f0 + u][j];
                }
                float v = (float)a;
                // tanh-approx GELU, matching activation="gelu" in PyTorch
                float c = 0.7978845608f * (v + 0.044715f*v*v*v);
                hid_t[u] = (data_t)(0.5f * v * (1.0f + hls::tanhf(c)));
            }
            FF_L2: for (int oc = 0; oc < VK_DIM; ++oc) {
#pragma HLS PIPELINE II=1
                acc_t a = acc2[oc];
                for (int j = 0; j < TILE_F; ++j) {
#pragma HLS UNROLL factor=UNROLL_F
                    a += (acc_t)hid_t[j] * (acc_t)w_l2[oc][f0 + j];
                }
                acc2[oc] = a;
            }
        }

        // ---- residual + LayerNorm, this token is done -------------------
        data_t y[VK_DIM];
#pragma HLS ARRAY_PARTITION variable=y cyclic factor=PF_TOK dim=1
        FN_Y: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
            y[c] = src[t][c] + (data_t)acc2[c];
        }
        acc_t mean = (acc_t)0;
        F_M: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
            mean += (acc_t)y[c];
        }
        float mu = (float)mean / (float)VK_DIM;
        acc_t var = (acc_t)0;
        F_V: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
            float d = (float)y[c] - mu;
            var += (acc_t)(d*d);
        }
        float inv_std = 1.0f / hls::sqrtf((float)var / (float)VK_DIM + 1e-5f);
        F_A: for (int c = 0; c < VK_DIM; ++c) {
#pragma HLS PIPELINE II=1
            dst[t][c] = (data_t)((((float)y[c] - mu) * inv_std)
                                 * (float)g_n2[c] + (float)be_n2[c]);
        }
    }
}

// ===========================================================================
// Stage 3: ConvTranspose2d(192,3,k=4,s=4). Stride == kernel, so each token
// paints a disjoint 4x4 output block -- no overlap-add, no accumulation
// hazard. PyTorch lays this weight out as [in_ch][out_ch][kh][kw].
// ===========================================================================
static void patch_unembed(const data_t src[N_TOK][VK_DIM],
                          const float *img_in, const float *w, float *img_out) {
    const float *uw = w;
    const float *ub = w + VK_DIM*C_OUT*VK_PATCH*VK_PATCH;

    UE_TOK: for (int t = 0; t < N_TOK; ++t) {
        int gy = t / GRID_W, gx = t % GRID_W;
        UE_PY: for (int py = 0; py < VK_PATCH; ++py) {
            UE_PX: for (int px = 0; px < VK_PATCH; ++px) {
                int iy = gy*VK_PATCH + py, ix = gx*VK_PATCH + px;
                UE_OC: for (int oc = 0; oc < C_OUT; ++oc) {
                    acc_t a = (acc_t)ub[oc];
                    // Pipeline the reduction itself, not the enclosing pixel
                    // loop: pipelining UE_PX forces a full unroll of 3x192
                    // MACs plus their DDR reads, measured at ~480k LUT.
                    UE_IC: for (int ic = 0; ic < VK_DIM; ++ic) {
#pragma HLS PIPELINE II=1
                        a += (acc_t)src[t][ic] *
                             (acc_t)uw[((ic*C_OUT + oc)*VK_PATCH + py)*VK_PATCH + px];
                    }
                    // Residual denoising, matching the DnCNN convention:
                    // the network predicts the NOISE, so subtract and clamp.
                    float v = img_in[(oc*IMG_H + iy)*IMG_W + ix] - (float)a;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 1.0f) v = 1.0f;
                    img_out[(oc*IMG_H + iy)*IMG_W + ix] = v;
                }
            }
        }
    }
}

// ===========================================================================
// Top level.
// ===========================================================================
void vit_denoiser_top(const float *img_in, const float *weights, float *img_out) {
#pragma HLS INTERFACE m_axi port=img_in  offset=slave bundle=gmem0 depth=37632
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem1 depth=1798000
#pragma HLS INTERFACE m_axi port=img_out offset=slave bundle=gmem2 depth=37632
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Vitis 2026.1 (clang-16) rejects '#pragma HLS' at file scope, so the
    // partition directives for the file-static buffers are declared here,
    // inside the top function. Semantics are identical.
#pragma HLS ARRAY_PARTITION variable=tokA cyclic factor=PF_TOK dim=2
#pragma HLS ARRAY_PARTITION variable=tokB cyclic factor=PF_TOK dim=2
#pragma HLS ARRAY_PARTITION variable=w_in  cyclic factor=PF_W dim=2
#pragma HLS ARRAY_PARTITION variable=w_out cyclic factor=PF_W dim=2
#pragma HLS ARRAY_PARTITION variable=w_l1  cyclic factor=PF_W dim=2
#pragma HLS ARRAY_PARTITION variable=w_l2  cyclic factor=PF_W dim=2
#pragma HLS ARRAY_PARTITION variable=hid_t cyclic factor=PF_W dim=1
#pragma HLS ARRAY_PARTITION variable=Kbuf cyclic factor=PF_TOK dim=2
#pragma HLS ARRAY_PARTITION variable=Vbuf cyclic factor=PF_TOK dim=2
#pragma HLS ARRAY_PARTITION variable=Qb cyclic factor=PF_TOK dim=2
#pragma HLS ARRAY_PARTITION variable=scores cyclic factor=PF_TOK dim=2

    const float *wp = weights;

    patch_embed(img_in, wp);
    wp += EMBED_W_SIZE;

    LAYERS: for (int l = 0; l < VK_DEPTH; ++l) {
        load_layer_weights(wp);
        wp += LAYER_W_SIZE;

        // tokA -> (attn) -> tokB -> (ffn) -> tokA
        attention_all(tokA, tokB);
        ffn_block(tokB, tokA, wp - LAYER_W_SIZE);
    }

    patch_unembed(tokA, img_in, wp, img_out);
}
