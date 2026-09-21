#include "denoiser_hls.h"
#include <cstdio>
#include <cmath>
int main() {
    const int NIN = C_IN*IMG_H*IMG_W, NOUT = C_OUT*IMG_H*IMG_W;
    static float in[NIN], out[NOUT], expd[NOUT];
    // Vitis runs csim from <proj>/<sol>/csim/build, four levels below the
    // project dir holding the .bin files; fall back to the bare name so this
    // also works when compiled and run directly.
    FILE *f = fopen("../../../../io_input.bin","rb");
    if(!f) f = fopen("io_input.bin","rb");
    if(!f){printf("no io_input.bin\n");return 2;}
    if(fread(in,sizeof(float),NIN,f)!=(size_t)NIN){printf("short read\n");return 2;}
    fclose(f);
    f = fopen("../../../../io_expected.bin","rb");
    if(!f) f = fopen("io_expected.bin","rb");
    if(!f){printf("no io_expected.bin\n");return 2;}
    if(fread(expd,sizeof(float),NOUT,f)!=(size_t)NOUT){printf("short read\n");return 2;}
    fclose(f);
    denoiser_top(in, out);
    double maxe=0, sume=0;
    for (int i=0;i<NOUT;++i){ double e=std::fabs((double)out[i]-(double)expd[i]); if(e>maxe)maxe=e; sume+=e; }
    printf("PREC=%d  max abs err vs float-PyTorch = %.4e   mean = %.4e\n", PREC, maxe, sume/NOUT);
#if PREC == 32
    if (maxe < 1e-3){ printf("PASS\n"); return 0; } printf("FAIL\n"); return 1;
#else
    printf("(fixed-point: this error magnitude is the precision-vs-accuracy result)\n"); return 0;
#endif
}
