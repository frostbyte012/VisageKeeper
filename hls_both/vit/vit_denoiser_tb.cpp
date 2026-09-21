#include "vit_denoiser_hls.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Golden-reference testbench, same contract as the DnCNN's denoiser_tb.cpp:
// at PREC=32 the kernel must match float PyTorch to 1e-3; at reduced precision
// the reported error IS the precision-vs-accuracy result for the paper.
int main() {
    const int NIN = C_IN*IMG_H*IMG_W, NOUT = C_OUT*IMG_H*IMG_W;
    static float in[NIN], out[NOUT], expd[NOUT];
    static float *w = (float*)malloc(sizeof(float) * TOTAL_W_SIZE);
    if (!w) { printf("malloc failed\n"); return 2; }

    // Vitis runs csim from <proj>/<sol>/csim/build; the .bin files sit four
    // levels up in the project dir. Fall back to the bare name so this also
    // works when the testbench is compiled and run directly.
    FILE *f = fopen("../../../../vit_weights.bin", "rb");
    if (!f) f = fopen("vit_weights.bin", "rb");
    if (!f) { printf("no vit_weights.bin -- run export_vit_weights.py\n"); return 2; }
    size_t got = fread(w, sizeof(float), TOTAL_W_SIZE, f); fclose(f);
    if (got != (size_t)TOTAL_W_SIZE) {
        printf("weights size mismatch: read %zu, expected %d\n", got, TOTAL_W_SIZE);
        return 2;
    }

    f = fopen("../../../../io_input.bin", "rb");
    if (!f) f = fopen("io_input.bin", "rb");
    if (!f) { printf("no io_input.bin\n"); return 2; }
    fread(in, sizeof(float), NIN, f); fclose(f);
    f = fopen("../../../../io_expected.bin", "rb");
    if (!f) f = fopen("io_expected.bin", "rb");
    if (!f) { printf("no io_expected.bin\n"); return 2; }
    fread(expd, sizeof(float), NOUT, f); fclose(f);

    vit_denoiser_top(in, w, out);

    double maxe = 0, sume = 0;
    for (int i = 0; i < NOUT; ++i) {
        double e = std::fabs((double)out[i] - (double)expd[i]);
        if (e > maxe) maxe = e;
        sume += e;
    }
    printf("PREC=%d  max abs err vs float-PyTorch = %.4e   mean = %.4e\n",
           PREC, maxe, sume/NOUT);
    free(w);
#if PREC == 32
    if (maxe < 1e-3) { printf("PASS\n"); return 0; }
    printf("FAIL\n"); return 1;
#else
    printf("(fixed-point: this error magnitude is the precision-vs-accuracy result)\n");
    return 0;
#endif
}
