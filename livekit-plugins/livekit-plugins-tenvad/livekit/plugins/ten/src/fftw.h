//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __FFTW_H__
#define __FFTW_H__

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
// Spectrum Storage Format definition:
// format1:  [Real-0, Real-Nyq, Real-1, Imag-1, Real-2, Imag-2, ...]
// format2:  [Real-0, Real-1, (-1)*Imag-1, Real-2, (-1)*Imag-2, ..., Real-Nyq]

// the following functions assume input and output spectrum to be stored in
// format2
void AUP_FFTW_r2c_256(float* in, float* out);
void AUP_FFTW_c2r_256(float* in, float* out);

void AUP_FFTW_c2r_512(float* in, float* out);
void AUP_FFTW_r2c_512(float* in, float* out);

void AUP_FFTW_r2c_1024(float* in, float* out);
void AUP_FFTW_c2r_1024(float* in, float* out);

void AUP_FFTW_r2c_2048(float* in, float* out);
void AUP_FFTW_c2r_2048(float* in, float* out);

void AUP_FFTW_r2c_4096(float* in, float* out);
void AUP_FFTW_c2r_4096(float* in, float* out);

// if direction == 0: format1->format2
// if direction == 1: format2->format1
void AUP_FFTW_InplaceTransf(int direction, int fftSz, float* inplaceTranfBuf);

void AUP_FFTW_RescaleFFTOut(int fftSz, float* inplaceBuf);
void AUP_FFTW_RescaleIFFTOut(int fftSz, float* inplaceBuf);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif  // __FFTW_H__
