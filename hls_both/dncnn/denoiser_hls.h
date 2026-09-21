#ifndef DENOISER_HLS_H
#define DENOISER_HLS_H
#define IMG_H 112
#define IMG_W 112
#define C_IN  3
#define C_MID 32
#define C_OUT 3
#ifndef PREC
#define PREC 32
#endif
#if PREC == 32
typedef float data_t; typedef float acc_t;
#elif PREC == 16
#include "ap_fixed.h"
typedef ap_fixed<16,4> data_t; typedef ap_fixed<32,8> acc_t;
#else
#include "ap_fixed.h"
typedef ap_fixed<8,2>  data_t; typedef ap_fixed<24,8> acc_t;
#endif
void denoiser_top(const float *img_in, float *img_out);
#endif
