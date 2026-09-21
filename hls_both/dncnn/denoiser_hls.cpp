#include "denoiser_hls.h"
#include <hls_stream.h>
#define WT float                 // float weight ROM -> fast compile; converted to data_t at runtime
#include "denoiser_weights.h"

template <int N> struct pix { data_t v[N]; };

// Conv layer (dataflow process), 3-row line buffer, same-pad 3x3.
// RELIABLE structure: one (ic,oc) pair per cycle (9 MACs), output-channel inner so
// acc[oc] recurrence distance = COUT. NO channel-dim partition and NO ic unroll --
// that combination is what makes the Vitis synthesizability checker hang. Keep it.

template <int CIN, int COUT, bool RELU>
static void conv_stage(hls::stream<pix<CIN> > &in, hls::stream<pix<COUT> > &out,
                       const WT *w, const WT *b) {
    data_t wl[COUT * CIN][9];
#pragma HLS ARRAY_PARTITION variable=wl complete dim=2
    for (int i = 0; i < COUT * CIN; ++i)
        for (int k = 0; k < 9; ++k) wl[i][k] = (data_t)w[i * 9 + k];

    data_t ring[3][IMG_W][CIN];
#pragma HLS ARRAY_PARTITION variable=ring complete dim=1
#pragma HLS ARRAY_PARTITION variable=ring cyclic factor=3 dim=2

    for (int iy = 0; iy <= IMG_H; ++iy) {
        if (iy < IMG_H) {
            for (int ix = 0; ix < IMG_W; ++ix) {
#pragma HLS PIPELINE II=1
                pix<CIN> p = in.read();
                for (int ic = 0; ic < CIN; ++ic) ring[iy % 3][ix][ic] = p.v[ic];
            }
        }
        int oy = iy - 1;
        if (oy >= 0) {
            bool top0 = (iy - 2 < 0);
            bool bot0 = (iy >= IMG_H);
            int rt = top0 ? 0 : (iy - 2) % 3;
            int rm = (iy - 1) % 3;
            int rb = iy % 3;
            for (int ox = 0; ox < IMG_W; ++ox) {
                acc_t acc[COUT];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1

                int oc = 0, ic = 0;
                for (int t = 0; t < CIN * COUT; ++t) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=acc type=inter dependent=false
                    acc_t d = (acc_t)0;
                    for (int ky = 0; ky < 3; ++ky) {
#pragma HLS UNROLL
                        int rr = (ky == 0) ? rt : (ky == 1) ? rm : rb;
                        bool zr = (ky == 0 && top0) || (ky == 2 && bot0);
                        for (int kx = 0; kx < 3; ++kx) {
#pragma HLS UNROLL
                            int ix = ox + kx - 1;
                            data_t v = (!zr && ix >= 0 && ix < IMG_W) ? ring[rr][ix][ic] : (data_t)0;
                            d += (acc_t)v * (acc_t)wl[oc * CIN + ic][ky * 3 + kx];
                        }
                    }
                    acc[oc] = (ic == 0) ? ((acc_t)b[oc] + d) : (acc[oc] + d);
                    if (++oc == COUT) { oc = 0; ic++; }
                }

                pix<COUT> o;
                for (int c = 0; c < COUT; ++c) {
#pragma HLS UNROLL
                    data_t r = (data_t)acc[c];
                    if (RELU && r < (data_t)0) r = (data_t)0;
                    o.v[c] = r;
                }
                out.write(o);
            }
        }
    }
}

static void load_stage(const float *img_in, hls::stream<pix<C_IN> > &s0,
                       hls::stream<pix<C_IN> > &sk) {
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
#pragma HLS PIPELINE II=1
            pix<C_IN> p;
            for (int c = 0; c < C_IN; ++c) p.v[c] = (data_t)img_in[(c * IMG_H + y) * IMG_W + x];
            s0.write(p);
            sk.write(p);
        }
}

static void store_stage(hls::stream<pix<C_OUT> > &sr, hls::stream<pix<C_IN> > &sk,
                        float *img_out) {
    for (int y = 0; y < IMG_H; ++y)
        for (int x = 0; x < IMG_W; ++x) {
#pragma HLS PIPELINE II=1
            pix<C_OUT> r = sr.read();
            pix<C_IN>  xin = sk.read();
            for (int c = 0; c < C_OUT; ++c) {
                data_t v = xin.v[c] - r.v[c];
                if (v < (data_t)0) v = (data_t)0;
                if (v > (data_t)1) v = (data_t)1;
                img_out[(c * IMG_H + y) * IMG_W + x] = (float)v;
            }
        }
}

void denoiser_top(const float *img_in, float *img_out) {
#pragma HLS INTERFACE m_axi port=img_in  offset=slave bundle=gmem0 depth=37632
#pragma HLS INTERFACE m_axi port=img_out offset=slave bundle=gmem1 depth=37632
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS DATAFLOW
    hls::stream<pix<C_IN> >  s0("s0"), sk("sk");
    hls::stream<pix<C_MID> > s1("s1"), s2("s2"), s3("s3");
    hls::stream<pix<C_OUT> > sr("sr");
#pragma HLS STREAM variable=sk depth=8192
#pragma HLS STREAM variable=s0 depth=512
#pragma HLS STREAM variable=s1 depth=512
#pragma HLS STREAM variable=s2 depth=512
#pragma HLS STREAM variable=s3 depth=512
#pragma HLS STREAM variable=sr depth=512

    load_stage(img_in, s0, sk);
    conv_stage<C_IN,  C_MID, true >(s0, s1, W0, B0);
    conv_stage<C_MID, C_MID, true >(s1, s2, W2, B2);
    conv_stage<C_MID, C_MID, true >(s2, s3, W4, B4);
    conv_stage<C_MID, C_OUT, false>(s3, sr, W6, B6);
    store_stage(sr, sk, img_out);
}