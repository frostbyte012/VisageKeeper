#ifndef VIT_DENOISER_HLS_H
#define VIT_DENOISER_HLS_H

// ---------------------------------------------------------------------------
// ViT denoiser (VisageKeeper) -- HLS header.
//
// Architecture mirrors frpure/defenses/vit_denoiser.py exactly:
//   Conv2d(3,192,k=4,s=4)  ->  +2D sin-cos pos  ->  4x TransformerEncoderLayer
//   (post-norm, 6 heads, FFN 768, GELU)  ->  ConvTranspose2d(192,3,k=4,s=4)
//
// KEY DIFFERENCE vs the DnCNN accelerator: the DnCNN has 0.02M weights, which
// fit in BRAM as a static ROM (denoiser_weights.h). The ViT has 1.80M weights
// = 3.5 MB at 16-bit, but the ZCU104 has only 11 Mb (1.375 MB) of BRAM. So
// weights CANNOT be resident. They are streamed from DDR per layer through an
// m_axi port and double-buffered against compute.
//
// Likewise the attention score matrix is N x N = 784 x 784 per head; at 16-bit
// that is 1.2 MB per head. It is never materialised -- attention is computed in
// row tiles of TILE_N queries, so only TILE_N x 784 scores live on chip.
// ---------------------------------------------------------------------------

#define IMG_H    112
#define IMG_W    112
#define C_IN     3
#define C_OUT    3

#define VK_PATCH    4
#define GRID_H   (IMG_H / VK_PATCH)          // 28
#define GRID_W   (IMG_W / VK_PATCH)          // 28
#define N_TOK    (GRID_H * GRID_W)        // 784
#define VK_DIM      192
#define VK_DEPTH    4
#define VK_HEADS    6
#define VK_DHEAD   (VK_DIM / VK_HEADS)            // 32
#define D_FF     768

// Query-row tile for attention. TILE_N x N_TOK scores stay on chip:
// 16 x 784 x 2 B = 24.5 KB. Raising this trades BRAM for fewer K/V re-reads.
#define TILE_N   16

// Array-partition factors. These are the main BRAM knob: HLS REPLICATES an
// array once per partition, so factor=16 on a 294 KB buffer can cost far more
// than 294 KB of block RAM. If csynth reports a BRAM overflow, halve these
// before shrinking the image -- measured DnCNN BRAM ran ~4x the raw array
// size for exactly this reason.
// Three separate knobs. Conflating them is what made the first build either
// overflow BRAM (all high) or run at 2.5 MAC/cycle (all low).
//
//   UNROLL_F : MACs issued per cycle in the inner reductions. Costs DSP/LUT,
//              which are plentiful (60/1728 DSP used at UNROLL_F=2).
//   PF_W     : partition factor on the REDUCTION axis of the weight arrays.
//              Must be >= UNROLL_F or the unrolled MACs stall on memory.
//              Applied to w_in/w_out/w_l1/w_l2 only -- these are small.
//   PF_TOK   : partition on the big N_TOK x DIM buffers. Kept LOW: these are
//              22 BRAM blocks each and replicate fastest. They are read
//              sequentially, so they do not need to match UNROLL_F.
#ifndef UNROLL_F
#define UNROLL_F 16
#endif
#ifndef PF_W
#define PF_W     16
#endif
#ifndef PF_TOK
#define PF_TOK   2
#endif

// K/V are held for ONE attention head at a time (784 x 32 instead of
// 784 x 192), cutting 588 KB to 98 KB. Cost is that the K/V projection is
// recomputed per head: +0.58 GFLOP, ~15%. Without this the design needs
// 2082 KB against the ZCU104's 1375 KB and does not fit.
#define KV_PER_HEAD 1

// FFN hidden-row tile. w_l1/w_l2 are 576 KB per layer at 16-bit, which does
// not fit alongside the token buffers, so the FFN streams in TILE_F row
// blocks: 128 x 192 x 2 B x 2 matrices = 96 KB resident.
#define TILE_F   128

// Precision select: 32 = float, 16 = ap_fixed<16,6>, 8 = ap_fixed<8,3>
// NOTE the integer widths differ from the DnCNN header. Transformer
// activations are NOT in [0,1]: LayerNorm outputs reach +/-6 and pre-softmax
// logits reach +/-20, so ap_fixed<16,4> (range +/-8) saturates. <16,6> gives
// +/-32 with 10 fractional bits.
#ifndef PREC
#define PREC 16
#endif

#if PREC == 32
typedef float data_t;
typedef float acc_t;
typedef float exp_t;
#elif PREC == 16
#include "ap_fixed.h"
typedef ap_fixed<16,6>  data_t;   // activations / weights
typedef ap_fixed<32,12> acc_t;    // MAC accumulator
typedef ap_fixed<24,10> exp_t;    // softmax numerator
#else
#include "ap_fixed.h"
typedef ap_fixed<8,3>   data_t;
typedef ap_fixed<24,10> acc_t;
typedef ap_fixed<20,8>  exp_t;
#endif

// Per-layer weight block laid out contiguously in DDR, in this order:
//   in_proj_w [3*VK_DIM][VK_DIM]   in_proj_b [3*VK_DIM]
//   out_proj_w[VK_DIM][VK_DIM]     out_proj_b[VK_DIM]
//   norm1_w[VK_DIM] norm1_b[VK_DIM]
//   lin1_w[D_FF][VK_DIM]        lin1_b[D_FF]
//   lin2_w[VK_DIM][D_FF]        lin2_b[VK_DIM]
//   norm2_w[VK_DIM] norm2_b[VK_DIM]
#define LAYER_W_SIZE  (3*VK_DIM*VK_DIM + 3*VK_DIM + VK_DIM*VK_DIM + VK_DIM + 2*VK_DIM \
                     + D_FF*VK_DIM + D_FF + VK_DIM*D_FF + VK_DIM + 2*VK_DIM)

// Byte offsets of each sublayer inside one layer's DDR block, so the FFN can
// seek to a row tile without loading the whole matrix.
#define OFF_LIN1_W  (3*VK_DIM*VK_DIM + 3*VK_DIM + VK_DIM*VK_DIM + VK_DIM + 2*VK_DIM)
#define OFF_LIN1_B  (OFF_LIN1_W + D_FF*VK_DIM)
#define OFF_LIN2_W  (OFF_LIN1_B + D_FF)
#define OFF_LIN2_B  (OFF_LIN2_W + VK_DIM*D_FF)
#define OFF_NORM2   (OFF_LIN2_B + VK_DIM)

// embed_w[VK_DIM][C_IN][VK_PATCH][VK_PATCH] embed_b[VK_DIM]
#define EMBED_W_SIZE  (VK_DIM*C_IN*VK_PATCH*VK_PATCH + VK_DIM)
// unembed_w[VK_DIM][C_OUT][VK_PATCH][VK_PATCH] unembed_b[C_OUT]
#define UNEMBED_W_SIZE (VK_DIM*C_OUT*VK_PATCH*VK_PATCH + C_OUT)

#define TOTAL_W_SIZE  (EMBED_W_SIZE + VK_DEPTH*LAYER_W_SIZE + UNEMBED_W_SIZE)

void vit_denoiser_top(const float *img_in, const float *weights, float *img_out);

#endif
