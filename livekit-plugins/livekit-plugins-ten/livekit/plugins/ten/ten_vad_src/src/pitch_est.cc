//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
// This file contains modified code derived from LPCNet (https://github.com/mozilla/LPCNet),
// specifically from the following functions:
//   - compute_frame_features() in lpcnet_enc.c
//   - process_superframe() in lpcnet_enc.c
//
// Original lpcnet_enc.c code LICENSE Text, licensed under the BSD-2-Clause License:
//   Copyright (c) 2017-2019 Mozilla
//
//   Redistribution and use in source and binary forms, with or without modification,
//   are permitted provided that the following conditions are met:
//
//   - Redistributions of source code must retain the above copyright notice, 
//     this list of conditions and the following disclaimer.
//
//   - Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
//   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//
// Original LPCNet LICENSE Text, licensed under the BSD-3-Clause License:
//   Copyright (c) 2017-2018, Mozilla
//   Copyright (c) 2007-2017, Jean-Marc Valin
//   Copyright (c) 2005-2017, Xiph.Org Foundation
//   Copyright (c) 2003-2004, Mark Borgerding
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions
//   are met:
//
//   - Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   - Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//  
//   - Neither the name of the Xiph.Org Foundation nor the names of its
//     contributors may be used to endorse or promote products derived from
//     this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION
//  OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//   

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "biquad.h"
#include "pitch_est.h"
#include "pitch_est_st.h"
#include "fftw.h"

// ==========================================================================================
// internal tools
// ==========================================================================================

static int AUP_PE_checkStatCfg(PE_StaticCfg* pCfg) {
  if (pCfg == NULL) {
    return -1;
  }

  if (pCfg->fftSz != 256 && pCfg->fftSz != 512 && pCfg->fftSz != 1024) {
    return -1;
  }
  if (pCfg->fftSz > AUP_PE_MAX_FFTSIZE) {
    return -1;
  }
  if (pCfg->anaWindowSz > pCfg->fftSz || pCfg->anaWindowSz < pCfg->hopSz) {
    return -1;
  }
  if (pCfg->hopSz != 64 && pCfg->hopSz != 80 && pCfg->hopSz != 128 &&
      pCfg->hopSz != 160 && pCfg->hopSz != 256 && pCfg->hopSz != 512) {
    return -1;
  }

  if (pCfg->useLPCPreFiltering != 0) {
    pCfg->useLPCPreFiltering = 1;
  }

  if (pCfg->procFs != 2000 && pCfg->procFs != 4000 && pCfg->procFs != 8000 &&
      pCfg->procFs != 16000) {
    pCfg->procFs = 4000;
  }

  return 0;
}

static int AUP_PE_checkDynamCfg(PE_DynamCfg* pCfg) {
  if (pCfg == NULL) {
    return -1;
  }

  pCfg->voicedThr = AUP_PE_MIN(2.0f, AUP_PE_MAX(pCfg->voicedThr, -1.0f));

  return 0;
}

static int AUP_PE_publishStaticCfg(PE_St* stHdl) {
  const PE_StaticCfg* pStatCfg;
  int idx, jdx;
  int hopSz;
  int excBufShiftLen;

  if (stHdl == NULL) {
    return -1;
  }
  pStatCfg = (const PE_StaticCfg*)(&(stHdl->stCfg));
  hopSz = (int)pStatCfg->hopSz;

  stHdl->nBins = ((int)(pStatCfg->fftSz >> 1)) + 1;
  stHdl->procResampleRate = AUP_PE_FS / (int)pStatCfg->procFs;
  stHdl->minPeriod = AUP_PE_MIN_PERIOD_16KHZ / stHdl->procResampleRate;
  stHdl->maxPeriod = AUP_PE_MAX_PERIOD_16KHZ / stHdl->procResampleRate;
  stHdl->difPeriod = stHdl->maxPeriod - stHdl->minPeriod;
  stHdl->inputResampleBufLen = hopSz * 2;  // give it a max. value
  stHdl->inputQLen = AUP_PE_MAX(AUP_PE_XCORR_TRAINING_OFFSET, hopSz) + hopSz;

  excBufShiftLen = (int)ceilf(hopSz / (float)stHdl->procResampleRate);
  stHdl->excBufLen = stHdl->maxPeriod + excBufShiftLen + 1;

  stHdl->nFeat = (int)ceilf(AUP_PE_FEAT_TIME_WINDOW * AUP_PE_FS /
                            ((float)hopSz * 1000.0f));
  stHdl->nFeat = AUP_PE_MIN(stHdl->nFeat, AUP_PE_FEAT_MAX_NFRM);
  stHdl->estDelay = 0;

  // publish DCT-table coeff.
  for (idx = 0; idx < AUP_PE_NB_BANDS; idx++) {
    for (jdx = 0; jdx < AUP_PE_NB_BANDS; jdx++) {
      stHdl->dct_table[idx * AUP_PE_NB_BANDS + jdx] =
          cosf((idx + .5f) * jdx * AUP_PE_PI / AUP_PE_NB_BANDS);
      if (jdx == 0) stHdl->dct_table[idx * AUP_PE_NB_BANDS + jdx] *= sqrtf(.5f);
    }
  }

  return 0;
}

static int AUP_PE_resetVariables(PE_St* stHdl) {
  // int nBins;
  int idx;

  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);

  stHdl->inputResampleBufIdx = 0;

  for (idx = 0; idx < AUP_PE_LPC_ORDER; idx++) {
    stHdl->lpc[idx] = 0;
    stHdl->pitch_mem[idx] = 0;
  }
  stHdl->pitch_filt = 0;

  memset(stHdl->tmpFeat, 0, sizeof(stHdl->tmpFeat));

  stHdl->xCorrOffsetIdx = 0;
  for (idx = 0; idx < (AUP_PE_FEAT_MAX_NFRM * 2); idx++) {
    stHdl->frmWeight[idx] = 0;
    stHdl->frmWeightNorm[idx] = 0;
  }

  stHdl->pitchMaxPathAll = 0;
  stHdl->bestPeriodEst = 0;

  stHdl->voiced = 0;
  stHdl->pitchEstResult = 0;  // label as no speech

  if (stHdl->procResampleRate != 1) {
    if (AUP_Biquad_init(stHdl->biquadIIRPtr) < 0) {
      return -1;
    }
  }

  return 0;
}

static int AUP_PE_dynamMemPrepare(PE_St* stHdl, void* memPtrExt,
                                  size_t memSize) {
  int idx;

  int inputResampleBufMemSize = 0;
  int inputQMemSize = 0;
  int alignedInMemSize = 0;
  int lpcFilterOutBufMemSize = 0;
  int excBufMemSize = 0;
  int excBufSqMemSize = 0;
  int xCorrInstMemSize = 0;
  int xCorrPerFeatMemSize = 0;
  int xCorrPerFeatTmpMemSize = 0;
  int pitchMaxPathRegPerRegMemSize = 0;
  int pitchPrevPerFeatMemSize = 0;
  int totalMemSize = 0;
  char* memPtr = NULL;

  inputResampleBufMemSize =
      AUP_PE_ALIGN8(sizeof(float) * stHdl->inputResampleBufLen);
  totalMemSize += inputResampleBufMemSize;

  inputQMemSize = AUP_PE_ALIGN8(sizeof(float) * stHdl->inputQLen);
  totalMemSize += inputQMemSize;

  alignedInMemSize = AUP_PE_ALIGN8(sizeof(float) * stHdl->stCfg.hopSz);
  totalMemSize += alignedInMemSize;

  lpcFilterOutBufMemSize = AUP_PE_ALIGN8(sizeof(float) * stHdl->stCfg.hopSz);
  totalMemSize += lpcFilterOutBufMemSize;

  excBufMemSize = AUP_PE_ALIGN8(sizeof(float) * stHdl->excBufLen);
  totalMemSize += excBufMemSize;
  excBufSqMemSize = excBufMemSize;
  totalMemSize += excBufSqMemSize;

  xCorrInstMemSize = AUP_PE_ALIGN8(sizeof(float) * (stHdl->maxPeriod));
  totalMemSize += xCorrInstMemSize;

  xCorrPerFeatMemSize = AUP_PE_ALIGN8(sizeof(float) * (stHdl->maxPeriod + 1));
  xCorrPerFeatTmpMemSize = xCorrPerFeatMemSize;
  totalMemSize +=
      (xCorrPerFeatMemSize + xCorrPerFeatTmpMemSize) * (stHdl->nFeat * 2);

  pitchMaxPathRegPerRegMemSize =
      AUP_PE_ALIGN8(sizeof(float) * (stHdl->maxPeriod));
  totalMemSize += pitchMaxPathRegPerRegMemSize * 2;

  pitchPrevPerFeatMemSize = AUP_PE_ALIGN8(sizeof(int) * (stHdl->maxPeriod));
  totalMemSize += pitchPrevPerFeatMemSize * (stHdl->nFeat * 2);

  // if no external memory provided, we are only profiling the memory
  // requirement
  if (memPtrExt == NULL) {
    return (totalMemSize);
  }

  // if required memory is more than provided, error
  if ((size_t)totalMemSize > memSize) {
    return -1;
  }

  memPtr = (char*)memPtrExt;

  stHdl->inputResampleBuf = (float*)memPtr;
  memPtr += inputResampleBufMemSize;

  stHdl->inputQ = (float*)memPtr;
  memPtr += inputQMemSize;

  stHdl->alignedIn = (float*)memPtr;
  memPtr += alignedInMemSize;

  stHdl->lpcFilterOutBuf = (float*)memPtr;
  memPtr += lpcFilterOutBufMemSize;

  stHdl->excBuf = (float*)memPtr;
  memPtr += excBufMemSize;

  stHdl->excBufSq = (float*)memPtr;
  memPtr += excBufSqMemSize;

  stHdl->xCorrInst = (float*)memPtr;
  memPtr += xCorrInstMemSize;

  for (idx = 0; idx < AUP_PE_FEAT_MAX_NFRM * 2; idx++) {
    stHdl->xCorr[idx] = NULL;
    stHdl->xCorrTmp[idx] = NULL;
    stHdl->pitchPrev[idx] = NULL;
  }
  for (idx = 0; idx < (stHdl->nFeat * 2); idx++) {
    stHdl->xCorr[idx] = (float*)memPtr;
    memPtr += xCorrPerFeatMemSize;

    stHdl->xCorrTmp[idx] = (float*)memPtr;
    memPtr += xCorrPerFeatTmpMemSize;

    stHdl->pitchPrev[idx] = (int*)memPtr;
    memPtr += pitchPrevPerFeatMemSize;
  }

  stHdl->pitchMaxPathReg[0] = (float*)memPtr;
  memPtr += pitchMaxPathRegPerRegMemSize;
  stHdl->pitchMaxPathReg[1] = (float*)memPtr;
  memPtr += pitchMaxPathRegPerRegMemSize;

  if (((int)(memPtr - (char*)memPtrExt)) > totalMemSize) {
    return -1;
  }

  return (totalMemSize);
}

static void AUP_PE_computeBandEnergy(const float* inBinPower,
                                     const int binPowNFFT,
                                     float bandE[AUP_PE_NB_BANDS]) {
  int i, j, bandSz;
  // float sum[NB_BANDS] = { 0 };
  float frac;
  float indexConvRate = 1.0;
  int indexOffset = 0, accIdx;
  int nBins = (binPowNFFT >> 1) + 1;

  indexConvRate = (float)binPowNFFT / AUP_PE_ASSUMED_FFT_4_BAND_ENG;
  for (i = 0; i < AUP_PE_NB_BANDS; i++) {
    bandE[i] = 0;
  }

  for (i = 0; i < AUP_PE_NB_BANDS - 1; i++) {
    bandSz = (int)roundf(
        (AUP_PE_BAND_START_INDEX[i + 1] - AUP_PE_BAND_START_INDEX[i]) *
        indexConvRate);  // WINDOW_SIZE_5MS;
    indexOffset = (int)roundf(AUP_PE_BAND_START_INDEX[i] *
                              indexConvRate);  // WINDOW_SIZE_5MS;

    for (j = 0; j < bandSz; j++) {
      frac = (float)j / bandSz;
      accIdx = AUP_PE_MIN(nBins - 1, (indexOffset + j));

      bandE[i] += (1 - frac) * inBinPower[accIdx];
      bandE[i + 1] += frac * inBinPower[accIdx];
    }
  }
  bandE[0] *= 2;
  bandE[AUP_PE_NB_BANDS - 1] *= 2;

  return;
}

static void AUP_PE_dct(const float DctTable[AUP_PE_NB_BANDS * AUP_PE_NB_BANDS],
                       const float* in, float* out) {
  int idx, j;
  float sum;
  float ratio = sqrtf(2.0f / AUP_PE_NB_BANDS);
  for (idx = 0; idx < AUP_PE_NB_BANDS; idx++) {
    sum = 0;
    for (j = 0; j < AUP_PE_NB_BANDS; j++) {
      sum += in[j] * DctTable[j * AUP_PE_NB_BANDS + idx];
    }
    out[idx] = sum * ratio;
  }
  return;
}

static void AUP_PE_idct(const float DctTable[AUP_PE_NB_BANDS * AUP_PE_NB_BANDS],
                        const float* in, float* out) {
  int idx, j;
  float sum;
  float ratio = sqrtf(2.0f / AUP_PE_NB_BANDS);
  for (idx = 0; idx < AUP_PE_NB_BANDS; idx++) {
    sum = 0;
    for (j = 0; j < AUP_PE_NB_BANDS; j++) {
      sum += in[j] * DctTable[idx * AUP_PE_NB_BANDS + j];
    }
    out[idx] = sum * ratio;
  }
  return;
}

static void AUP_PE_interp_band_gain(const int nBins,
                                    const float bandE[AUP_PE_NB_BANDS],
                                    float* g) {
  int idx, j, bandSz;
  float indexConvRate = 1.0f;
  int fftSz = (nBins - 1) * 2;
  int indexOffset = 0, accIdx;
  float frac;

  indexConvRate = ((float)fftSz) / AUP_PE_ASSUMED_FFT_4_BAND_ENG;
  memset(g, 0, sizeof(float) * nBins);

  for (idx = 0; idx < AUP_PE_NB_BANDS - 1; idx++) {
    bandSz = (int)roundf(
        (AUP_PE_BAND_START_INDEX[idx + 1] - AUP_PE_BAND_START_INDEX[idx]) *
        indexConvRate);  // WINDOW_SIZE_5MS;
    indexOffset = (int)roundf(AUP_PE_BAND_START_INDEX[idx] *
                              indexConvRate);  // WINDOW_SIZE_5MS;

    for (j = 0; j < bandSz; j++) {
      frac = (float)j / bandSz;
      accIdx = AUP_PE_MIN(nBins - 1, (indexOffset + j));

      g[accIdx] = (1 - frac) * bandE[idx] + frac * bandE[idx + 1];
    }
  }

  return;
}

// ac: in:  [0...p] autocorrelation values
// p: in: buffer length of _lpc and rc
// _lpc: out: [0...p-1] LPC coefficients
static float AUP_PE_celt_lpc(const float* ac, const int p, float* _lpc,
                             float* rc) {
  int i, j;
  float r;
  float error = ac[0];
  float* lpc = _lpc;
  float rr;
  float tmp1, tmp2;

  // RNN_CLEAR(lpc, p);
  memset(lpc, 0, sizeof(float) * p);
  // RNN_CLEAR(rc, p);
  memset(rc, 0, sizeof(float) * p);

  if (ac[0] != 0) {
    for (i = 0; i < p; i++) {
      /* Sum up this iteration's reflection coefficient */
      rr = 0;
      for (j = 0; j < i; j++) rr += lpc[j] * ac[i - j];
      rr += ac[i + 1];
      r = (-rr) / error;
      rc[i] = r;
      /*  Update LPC coefficients and total error */
      lpc[i] = r;
      for (j = 0; j<(i + 1)>> 1; j++) {
        tmp1 = lpc[j];
        tmp2 = lpc[i - 1 - j];
        lpc[j] = tmp1 + (r * tmp2);
        lpc[i - 1 - j] = tmp2 + (r * tmp1);
      }

      error = error - (r * r * error);
      /* Bail out once we get 30 dB gain */

      if (error < .001f * ac[0]) break;
    }
  }

  return error;
}

static float AUP_PE_lpc_from_bands(const int windowSz, const int nBins,
                                   const float Ex[AUP_PE_NB_BANDS],
                                   float lpc[AUP_PE_LPC_ORDER]) {
  int i;
  float e;
  float ac[AUP_PE_LPC_ORDER + 1] = {0};
  float rc[AUP_PE_LPC_ORDER] = {0};
  float Xr[AUP_PE_MAX_NBINS] = {0};
  float X_auto[AUP_PE_MAX_FFTSIZE + 4] = {0};
  float x_auto[AUP_PE_MAX_FFTSIZE + 4] = {0};
  float DC0_BIAS;
  int fftSz = (nBins - 1) * 2;

  AUP_PE_interp_band_gain(nBins, Ex, Xr);
  Xr[nBins - 1] = 0;  // remove nyquist freq.

  // RNN_CLEAR(X_auto, FREQ_SIZE);
  X_auto[0] = Xr[0];  // reformat as complex spectrum data
  X_auto[1] = Xr[nBins - 1];
  for (i = 1; i < (nBins - 1); i++) {
    X_auto[i << 1] = Xr[i];  // give value to its real part
  }                          // leave all the imaginary part as 0

  // inverse_transform(x_auto, X_auto); // IFFT, transform back to time domain
  // (X_auto -> x_auto)
  AUP_FFTW_InplaceTransf(0, fftSz, X_auto);
  if (fftSz == 256) {
    AUP_FFTW_c2r_256(X_auto, x_auto);
  } else if (fftSz == 512) {
    AUP_FFTW_c2r_512(X_auto, x_auto);
  } else if (fftSz == 1024) {
    AUP_FFTW_c2r_1024(X_auto, x_auto);
  }
  AUP_FFTW_RescaleIFFTOut(fftSz, x_auto);

  for (i = 0; i < (AUP_PE_LPC_ORDER + 1);
       i++) {  // take only the first LPC_ORDER + 1 coeff.
    ac[i] = x_auto[i];
  }

  // -40 dB noise floor
  DC0_BIAS = (windowSz / 12 / 38.0f);

  ac[0] += ac[0] * 1e-4f + DC0_BIAS;
  // Lag windowing
  for (i = 1; i < (AUP_PE_LPC_ORDER + 1); i++) {
    ac[i] *= (1 - 6e-5f * i * i);
  }

  e = AUP_PE_celt_lpc(ac, AUP_PE_LPC_ORDER, lpc, rc);

  return (e);
}

// lpc_from_cepstrum
static float AUP_PE_lpcCompute(
    const int windowSz, const int nBins,
    const float DctTable[AUP_PE_NB_BANDS * AUP_PE_NB_BANDS],
    const float* cepstrum, float* lpc) {
  int i;
  float Ex[AUP_PE_NB_BANDS] = {0};
  float tmp[AUP_PE_NB_BANDS] = {0};
  float errValue = 0;

  // RNN_COPY(tmp, cepstrum, NB_BANDS);
  memcpy(tmp, cepstrum, sizeof(float) * AUP_PE_NB_BANDS);

  AUP_PE_idct(DctTable, tmp, Ex);  // idct(Ex, tmp);
  for (i = 0; i < AUP_PE_NB_BANDS; i++) {
    Ex[i] = powf(10.f, Ex[i]) * AUP_PE_BAND_LPC_COMP[i];
  }

  errValue = AUP_PE_lpc_from_bands(windowSz, nBins, Ex, lpc);

  return (errValue);
}

static void AUP_PE_xcorr_kernel(const float* x, const float* y, float sum[4],
                                int len) {
  int j;
  float y_0, y_1, y_2, y_3;
  y_3 = 0; /* gcc doesn't realize that y_3 can't be used uninitialized */
  y_0 = *y++;
  y_1 = *y++;
  y_2 = *y++;
  for (j = 0; j < len - 3; j += 4) {
    float tmp;
    tmp = *x++;
    y_3 = *y++;
    sum[0] += tmp * y_0;
    sum[1] += tmp * y_1;
    sum[2] += tmp * y_2;
    sum[3] += tmp * y_3;
    tmp = *x++;
    y_0 = *y++;
    sum[0] += tmp * y_1;
    sum[1] += tmp * y_2;
    sum[2] += tmp * y_3;
    sum[3] += tmp * y_0;
    tmp = *x++;
    y_1 = *y++;
    sum[0] += tmp * y_2;
    sum[1] += tmp * y_3;
    sum[2] += tmp * y_0;
    sum[3] += tmp * y_1;
    tmp = *x++;
    y_2 = *y++;
    sum[0] += tmp * y_3;
    sum[1] += tmp * y_0;
    sum[2] += tmp * y_1;
    sum[3] += tmp * y_2;
  }
  if (j++ < len) {
    float tmp = *x++;
    y_3 = *y++;
    sum[0] += tmp * y_0;
    sum[1] += tmp * y_1;
    sum[2] += tmp * y_2;
    sum[3] += tmp * y_3;
  }
  if (j++ < len) {
    float tmp = *x++;
    y_0 = *y++;
    sum[0] += tmp * y_1;
    sum[1] += tmp * y_2;
    sum[2] += tmp * y_3;
    sum[3] += tmp * y_0;
  }
  if (j < len) {
    float tmp = *x++;
    y_1 = *y++;
    sum[0] += tmp * y_2;
    sum[1] += tmp * y_3;
    sum[2] += tmp * y_0;
    sum[3] += tmp * y_1;
  }
  return;
}

static float AUP_PE_celt_inner_prod(const float* x, const float* y, int N) {
  int i;
  float xy = 0;
  for (i = 0; i < N; i++) {
    xy += (x[i] * y[i]);
  }

  return (xy);
}

static void AUP_PE_MvingXCorr(int corrWindowLen, int corrShiftTimes,
                              const float* refIn, const float* yInToShift,
                              float* xcorr) {
  /* Unrolled version of the pitch correlation -- runs faster on x86 and ARM */
  int i;
  float tmp;

  for (i = 0; i < corrShiftTimes - 3; i += 4) {
    float sum[4] = {0, 0, 0, 0};
    AUP_PE_xcorr_kernel(refIn, yInToShift + i, sum, corrWindowLen);
    xcorr[i] = sum[0];
    xcorr[i + 1] = sum[1];
    xcorr[i + 2] = sum[2];
    xcorr[i + 3] = sum[3];
  }
  /* In case corrShiftTimes isn't a multiple of 4, do non-unrolled version. */
  for (; i < corrShiftTimes; i++) {
    tmp = AUP_PE_celt_inner_prod(refIn, yInToShift + i, corrWindowLen);
    xcorr[i] = tmp;
  }
  return;
}

// ==========================================================================================
// public APIs
// ==========================================================================================

int AUP_PE_create(void** stPtr) {
  PE_St* tmpPtr;

  if (stPtr == NULL) {
    return -1;
  }

  *stPtr = (void*)malloc(sizeof(PE_St));
  if (*stPtr == NULL) {
    return -1;
  }
  memset(*stPtr, 0, sizeof(PE_St));

  tmpPtr = (PE_St*)(*stPtr);

  tmpPtr->dynamMemPtr = NULL;
  tmpPtr->dynamMemSize = 0;

  if (AUP_Biquad_create(&(tmpPtr->biquadIIRPtr)) < 0 ||
      tmpPtr->biquadIIRPtr == NULL) {
    return -1;
  }

  tmpPtr->stCfg.fftSz = 1024;
  tmpPtr->stCfg.anaWindowSz = 768;
  tmpPtr->stCfg.hopSz = 256;
  tmpPtr->stCfg.useLPCPreFiltering = 1;
  tmpPtr->stCfg.procFs = 4000;  // 4KHz resampling rate

  tmpPtr->dynamCfg.voicedThr = 0.4f;

  return 0;
}

int AUP_PE_destroy(void** stPtr) {
  PE_St* stHdl;

  if (stPtr == NULL) {
    return 0;
  }

  stHdl = (PE_St*)(*stPtr);
  if (stHdl == NULL) {
    return 0;
  }

  if (stHdl->dynamMemPtr != NULL) {
    free(stHdl->dynamMemPtr);
  }
  stHdl->dynamMemPtr = NULL;

  if (stHdl->biquadIIRPtr != NULL) {
    AUP_Biquad_destroy(&(stHdl->biquadIIRPtr));
  }

  free(stHdl);
  (*stPtr) = NULL;

  return 0;
}

int AUP_PE_memAllocate(void* stPtr, const PE_StaticCfg* pCfg) {
  PE_St* stHdl = NULL;
  PE_StaticCfg localStCfg;
  Biquad_StaticCfg biquadStCfg = {0, 0, 0, {0}, {0}, 0};
  int idx;
  int totalMemSize = 0;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (PE_St*)(stPtr);

  memcpy(&localStCfg, pCfg, sizeof(PE_StaticCfg));
  if (AUP_PE_checkStatCfg(&localStCfg) < 0) {
    return -1;
  }

  memcpy(&(stHdl->stCfg), &localStCfg, sizeof(PE_StaticCfg));

  // publish internal static configuration registers
  if (AUP_PE_publishStaticCfg(stHdl) < 0) {
    return -1;
  }

  // check memory requirement
  totalMemSize = AUP_PE_dynamMemPrepare(stHdl, NULL, 0);
  if (totalMemSize < 0) {
    return -1;
  }

  // allocate dynamic memory
  if ((size_t)totalMemSize > stHdl->dynamMemSize) {
    if (stHdl->dynamMemPtr != NULL) {
      free(stHdl->dynamMemPtr);
      stHdl->dynamMemSize = 0;
    }
    stHdl->dynamMemPtr = malloc(totalMemSize);
    if (stHdl->dynamMemPtr == NULL) {
      return -1;
    }
    stHdl->dynamMemSize = totalMemSize;
  }
  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);

  // setup the pointers/variable
  if (AUP_PE_dynamMemPrepare(stHdl, stHdl->dynamMemPtr, stHdl->dynamMemSize) <
      0) {
    return -1;
  }

  if (AUP_Biquad_getStaticCfg(stHdl->biquadIIRPtr, &biquadStCfg) < 0) {
    return -1;
  }
  biquadStCfg.maxNSample = stHdl->stCfg.hopSz;
  if (stHdl->procResampleRate != 1) {
    biquadStCfg.nsect = AUP_PE_LOWPSS_NSEC;
    if (stHdl->stCfg.procFs == 2000) {
      biquadStCfg.G = AUP_PE_G_2KHZ;
      for (idx = 0; idx < biquadStCfg.nsect; idx++) {
        biquadStCfg.B[idx] = AUP_PE_B_2KHZ[idx];
        biquadStCfg.A[idx] = AUP_PE_A_2KHZ[idx];
      }
    } else if (stHdl->stCfg.procFs == 4000) {
      biquadStCfg.G = AUP_PE_G_4KHZ;
      for (idx = 0; idx < biquadStCfg.nsect; idx++) {
        biquadStCfg.B[idx] = AUP_PE_B_4KHZ[idx];
        biquadStCfg.A[idx] = AUP_PE_A_4KHZ[idx];
      }
    } else if (stHdl->stCfg.procFs == 8000) {
      biquadStCfg.G = AUP_PE_G_8KHZ;
      for (idx = 0; idx < biquadStCfg.nsect; idx++) {
        biquadStCfg.B[idx] = AUP_PE_B_8KHZ[idx];
        biquadStCfg.A[idx] = AUP_PE_A_8KHZ[idx];
      }
    }
  } else {
    biquadStCfg.nsect = -1;
  }
  if (AUP_Biquad_memAllocate(stHdl->biquadIIRPtr, &biquadStCfg) < 0) {
    return -1;
  }

  return 0;
}

int AUP_PE_init(void* stPtr) {
  PE_St* stHdl;

  if (stPtr == NULL) {
    return -1;
  }
  stHdl = (PE_St*)(stPtr);

  if (AUP_PE_resetVariables(stHdl) < 0) {
    return -1;
  }

  return 0;
}

int AUP_PE_setDynamCfg(void* stPtr, const PE_DynamCfg* pCfg) {
  PE_St* stHdl;
  PE_DynamCfg localCfg;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }

  memcpy(&localCfg, pCfg, sizeof(PE_DynamCfg));
  if (AUP_PE_checkDynamCfg(&localCfg) < 0) {
    return -1;
  }

  stHdl = (PE_St*)(stPtr);

  memcpy(&(stHdl->dynamCfg), &localCfg, sizeof(PE_DynamCfg));

  return 0;
}

int AUP_PE_getStaticCfg(const void* stPtr, PE_StaticCfg* pCfg) {
  const PE_St* stHdl;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (const PE_St*)(stPtr);

  memcpy(pCfg, &(stHdl->stCfg), sizeof(PE_StaticCfg));

  return 0;
}

int AUP_PE_getDynamCfg(const void* stPtr, PE_DynamCfg* pCfg) {
  const PE_St* stHdl;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (const PE_St*)(stPtr);

  memcpy(pCfg, &(stHdl->dynamCfg), sizeof(PE_DynamCfg));

  return 0;
}

int AUP_PE_getAlgDelay(const void* stPtr, int* delayInFrms) {
  const PE_St* stHdl;

  if (stPtr == NULL || delayInFrms == NULL) {
    return -1;
  }
  stHdl = (const PE_St*)(stPtr);

  *delayInFrms = stHdl->estDelay;

  return 0;
}

int AUP_PE_proc(void* stPtr, const PE_InputData* pIn, PE_OutputData* pOut) {
  PE_St* stHdl = NULL;
  Biquad_InputData bqInData = {0, 0, 0};
  Biquad_OutputData bqOutData = {0};
  int nBins, fftSz, hopSz, idx, jdx, sub, offset, tmpInt, xcorrAccIdx;
  float bandPow[AUP_PE_NB_BANDS] = {0};  // Ex
  float Ly[AUP_PE_NB_BANDS] = {0};
  float follow, lpcErr, logMax;
  float energy0, slidWinSum, tmpDenom = 0, maxTrackReg = 0, maxPathReg = 0;
  float frmCorr = 0;  // frmCorrCorrection = 0;
  const float* startPtr = NULL;
  const float* refSeqPtr = NULL;
  const float* mvSeqPtr = NULL;
  int CORR_HALF_HOPSZ, SIDXT, XCIdx;
  int bestPeriodEstLocal[AUP_PE_TOTAL_NFEAT * 2] = {0};
  float w, sx = 0, sxx = 0, sxy = 0, sy = 0, sw = 0;
  float bestA = 0, bestB = 0;
  float estimatedPeriod;

  if (stPtr == NULL || pIn == NULL || pIn->timeSignal == NULL) {
    return -1;
  }
  stHdl = (PE_St*)(stPtr);

  nBins = (int)(stHdl->nBins);
  fftSz = (int)(stHdl->stCfg.fftSz);
  hopSz = (int)(stHdl->stCfg.hopSz);
  CORR_HALF_HOPSZ = hopSz / (stHdl->procResampleRate * 2);

  if (pIn->hopSz != (int)stHdl->stCfg.hopSz) {
    return -1;
  }

  if (stHdl->stCfg.useLPCPreFiltering == 1 && pIn->inBinPow == NULL) {
    return -1;
  }
  if (stHdl->stCfg.useLPCPreFiltering == 1 && pIn->nBins != stHdl->nBins) {
    return -1;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Feature Pre-Calculation .... from compute_frame_features of lpcnet_enc.cc
  //////////////////////////////////////////////////////////////////////////////////////
  if (stHdl->stCfg.useLPCPreFiltering == 1) {
    // first. generate features and pre-raw information ...
    AUP_PE_computeBandEnergy(pIn->inBinPow, fftSz, bandPow);
    logMax = -2.0f;
    follow = -2.0f;
    for (idx = 0; idx < AUP_PE_NB_BANDS; idx++) {
      Ly[idx] = log10f(1e-2f + bandPow[idx]);  // Ex
      Ly[idx] = AUP_PE_MAX(logMax - 8.0f, AUP_PE_MAX(follow - 2.5f, Ly[idx]));
      logMax = AUP_PE_MAX(logMax, Ly[idx]);

      follow = AUP_PE_MAX(follow - 2.5f, Ly[idx]);
    }

    AUP_PE_dct(stHdl->dct_table, Ly, stHdl->tmpFeat);

    lpcErr = AUP_PE_lpcCompute((int)(stHdl->stCfg.anaWindowSz), nBins,
                               stHdl->dct_table, stHdl->tmpFeat, stHdl->lpc);

    memmove(stHdl->inputQ, stHdl->inputQ + hopSz,
            sizeof(float) * (stHdl->inputQLen - hopSz));
    memcpy(&(stHdl->inputQ[stHdl->inputQLen - hopSz]), pIn->timeSignal,
           sizeof(float) * hopSz);
    // then, take part out into alignedIn for later correlation calculation
    offset =
        AUP_PE_MAX(0, stHdl->inputQLen - hopSz - AUP_PE_XCORR_TRAINING_OFFSET);
    memcpy(stHdl->alignedIn, stHdl->inputQ + offset, sizeof(float) * hopSz);

    for (idx = 0; idx < hopSz; idx++) {
      // FIR LPC filtering .....
      slidWinSum = stHdl->alignedIn[idx];
      for (jdx = 0; jdx < AUP_PE_LPC_ORDER; jdx++) {
        slidWinSum += stHdl->lpc[jdx] * stHdl->pitch_mem[jdx];
      }

      memmove(stHdl->pitch_mem + 1, stHdl->pitch_mem,
              sizeof(float) * (AUP_PE_LPC_ORDER - 1));
      stHdl->pitch_mem[0] =
          stHdl->alignedIn[idx];  // push the latest base-sample into the tail
                                  // of FIFO

      stHdl->lpcFilterOutBuf[idx] = slidWinSum + 0.7f * stHdl->pitch_filt;
      stHdl->pitch_filt = slidWinSum;
    }

    if (stHdl->procResampleRate != 1) {
      // resample of lpcFilterOutBuf
      bqInData.nsamples = (size_t)hopSz;
      bqInData.samplesPtr = (const void*)(stHdl->lpcFilterOutBuf);
      bqInData.sampleType = 1;
      bqOutData.outputBuff =
          (void*)(stHdl->inputResampleBuf + stHdl->inputResampleBufIdx);
      if (AUP_Biquad_proc(stHdl->biquadIIRPtr, &bqInData, &bqOutData) < 0) {
        return -1;
      }
      tmpInt = stHdl->inputResampleBufIdx;
      for (idx = tmpInt; idx < (tmpInt + hopSz);
           idx += stHdl->procResampleRate) {
        stHdl->inputResampleBuf[stHdl->inputResampleBufIdx] =
            stHdl->inputResampleBuf[idx];
        stHdl->inputResampleBufIdx++;
      }
      // update the excBuf ....
      tmpInt = stHdl->inputResampleBufIdx;
      memmove(stHdl->excBuf, stHdl->excBuf + tmpInt,
              sizeof(float) * (stHdl->excBufLen - tmpInt));
      memcpy(stHdl->excBuf + (stHdl->excBufLen - tmpInt),
             stHdl->inputResampleBuf, sizeof(float) * tmpInt);
      stHdl->inputResampleBufIdx = 0;
    } else {
      tmpInt = hopSz;
      memmove(stHdl->excBuf, stHdl->excBuf + tmpInt,
              sizeof(float) * (stHdl->excBufLen - tmpInt));
      memcpy(stHdl->excBuf + (stHdl->excBufLen - tmpInt),
             stHdl->lpcFilterOutBuf, sizeof(float) * tmpInt);
    }

  } else {
    if (stHdl->procResampleRate != 1) {
      // resample of lpcFilterOutBuf
      bqInData.nsamples = (size_t)hopSz;
      bqInData.samplesPtr = (const void*)(pIn->timeSignal);
      bqInData.sampleType = 1;
      bqOutData.outputBuff =
          (void*)(stHdl->inputResampleBuf + stHdl->inputResampleBufIdx);
      if (AUP_Biquad_proc(stHdl->biquadIIRPtr, &bqInData, &bqOutData) < 0) {
        return -1;
      }
      tmpInt = stHdl->inputResampleBufIdx;
      for (idx = tmpInt; idx < (tmpInt + hopSz);
           idx += stHdl->procResampleRate) {
        stHdl->inputResampleBuf[stHdl->inputResampleBufIdx] =
            stHdl->inputResampleBuf[idx];
        stHdl->inputResampleBufIdx++;
      }

      // update the excBuf ....
      tmpInt = stHdl->inputResampleBufIdx;
      memmove(stHdl->excBuf, stHdl->excBuf + tmpInt,
              sizeof(float) * (stHdl->excBufLen - tmpInt));
      memcpy(stHdl->excBuf + (stHdl->excBufLen - tmpInt),
             stHdl->inputResampleBuf, sizeof(float) * tmpInt);
      stHdl->inputResampleBufIdx = 0;
    } else {
      tmpInt = hopSz;
      memmove(stHdl->excBuf, stHdl->excBuf + tmpInt,
              sizeof(float) * (stHdl->excBufLen - tmpInt));
      memcpy(stHdl->excBuf + (stHdl->excBufLen - tmpInt), pIn->timeSignal,
             sizeof(float) * tmpInt);
    }
  }

  // prepare for cross-correlation computation ....
  for (idx = 0; idx < stHdl->excBufLen; idx++) {
    stHdl->excBufSq[idx] = (stHdl->excBuf[idx] * stHdl->excBuf[idx]);
  }

  // shift the frmWeight queue to left space for this new frame
  for (idx = 0; idx < (stHdl->nFeat - 1); idx++) {
    stHdl->frmWeight[2 * (idx)] = stHdl->frmWeight[2 * (idx + 1)];
    stHdl->frmWeight[2 * (idx) + 1] = stHdl->frmWeight[2 * (idx + 1) + 1];
  }

  // do the cross-correlation .....
  for (sub = 0; sub < 2; sub++) {
    xcorrAccIdx = 2 * (stHdl->xCorrOffsetIdx) + sub;
    offset = sub * CORR_HALF_HOPSZ;

    refSeqPtr = stHdl->excBuf + (stHdl->maxPeriod + offset);
    mvSeqPtr = stHdl->excBuf + offset;
    AUP_PE_MvingXCorr(CORR_HALF_HOPSZ, stHdl->maxPeriod, refSeqPtr, mvSeqPtr,
                      stHdl->xCorrInst);

    energy0 = 0;
    startPtr = stHdl->excBufSq + (stHdl->maxPeriod + offset);
    for (idx = 0; idx < CORR_HALF_HOPSZ; idx++) {
      energy0 += startPtr[idx];
    }
    stHdl->frmWeight[2 * (stHdl->nFeat - 1) + sub] = energy0;

    slidWinSum = 0;
    startPtr = stHdl->excBufSq + offset;
    for (idx = 0; idx < CORR_HALF_HOPSZ; idx++) {
      slidWinSum += startPtr[idx];
    }

    // special hanlding for the 0th element
    tmpDenom = AUP_PE_MAX(1e-12f, slidWinSum + (1 + energy0));
    stHdl->xCorr[xcorrAccIdx][0] = 2 * stHdl->xCorrInst[0] / tmpDenom;

    for (idx = 1; idx < stHdl->maxPeriod; idx++) {
      // update the slidWinSum
      slidWinSum =
          AUP_PE_MAX(0, slidWinSum - stHdl->excBufSq[offset + idx - 1]);
      slidWinSum += stHdl->excBufSq[offset + idx + CORR_HALF_HOPSZ - 1];

      tmpDenom = AUP_PE_MAX(1e-12f, slidWinSum + (1 + energy0));
      stHdl->xCorr[xcorrAccIdx][idx] = 2 * stHdl->xCorrInst[idx] / tmpDenom;
    }

    // shrink/sharpen the values in xCorr array ...
    for (idx = 0; idx < (stHdl->maxPeriod - 2 * stHdl->minPeriod); idx++) {
      tmpDenom = stHdl->xCorr[xcorrAccIdx][(stHdl->maxPeriod + idx) / 2];
      tmpDenom = AUP_PE_MAX(
          tmpDenom,
          stHdl->xCorr[xcorrAccIdx][(stHdl->maxPeriod + idx + 2) / 2]);
      tmpDenom = AUP_PE_MAX(
          tmpDenom,
          stHdl->xCorr[xcorrAccIdx][(stHdl->maxPeriod + idx - 1) / 2]);

      if (stHdl->xCorr[xcorrAccIdx][idx] < (tmpDenom * 1.1f))
        stHdl->xCorr[xcorrAccIdx][idx] *= 0.8f;
    }
  }
  stHdl->xCorrOffsetIdx++;
  if (stHdl->xCorrOffsetIdx >= stHdl->nFeat) {
    stHdl->xCorrOffsetIdx = 0;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Pitch Estimation .... from process_superframe of lpcnet_enc.cc
  //////////////////////////////////////////////////////////////////////////////////////
  slidWinSum = 1e-15f;
  for (sub = 0; sub < (stHdl->nFeat * 2); sub++) {
    slidWinSum += stHdl->frmWeight[sub];
  }
  for (sub = 0; sub < (stHdl->nFeat * 2); sub++) {
    stHdl->frmWeightNorm[sub] =
        stHdl->frmWeight[sub] * ((stHdl->nFeat * 2) / slidWinSum);
  }

  // copy xCorr to xCorrTmp, so that later-on we can modify the content in
  // xCorrTmp without impacting next hop's processing
  for (idx = 0; idx < (stHdl->nFeat * 2); idx++) {
    memcpy(stHdl->xCorrTmp[idx], stHdl->xCorr[idx],
           sizeof(float) * (stHdl->maxPeriod + 1));
  }

  // shift pitchPrev buffer to left space for this new frame's result
  for (sub = 0; sub < (stHdl->nFeat * 2 - 2); sub += 2) {
    memcpy(stHdl->pitchPrev[sub], stHdl->pitchPrev[sub + 2],
           sizeof(int) * stHdl->maxPeriod);
    memcpy(stHdl->pitchPrev[sub + 1], stHdl->pitchPrev[sub + 3],
           sizeof(int) * stHdl->maxPeriod);
  }
  for (sub = (stHdl->nFeat * 2 - 2); sub < (stHdl->nFeat * 2); sub++) {
    XCIdx = sub + (stHdl->xCorrOffsetIdx * 2);
    if (XCIdx >= (2 * stHdl->nFeat)) {
      XCIdx -= (2 * stHdl->nFeat);
    }

    for (idx = 0; idx < stHdl->difPeriod; idx++) {
      maxTrackReg = stHdl->pitchMaxPathAll - 1e10f;
      stHdl->pitchPrev[sub][idx] = stHdl->bestPeriodEst;

      SIDXT = AUP_PE_MIN(0, 4 - idx);
      for (jdx = SIDXT; jdx <= 4 && (idx + jdx) < stHdl->difPeriod; jdx++) {
        tmpDenom = stHdl->pitchMaxPathReg[0][idx + jdx] -
                   (AUP_PE_PITCHMAXPATH_W * abs(jdx) * abs(jdx));
        if (tmpDenom > maxTrackReg) {
          maxTrackReg = tmpDenom;
          stHdl->pitchPrev[sub][idx] = idx + jdx;
        }
      }

      // store the max search result into pitch_max_path[1][...]
      stHdl->pitchMaxPathReg[1][idx] =
          maxTrackReg + stHdl->frmWeightNorm[sub] * stHdl->xCorrTmp[XCIdx][idx];
    }

    maxPathReg = -1e15f;
    tmpInt = 0;
    for (idx = 0; idx < stHdl->difPeriod; idx++) {
      if (stHdl->pitchMaxPathReg[1][idx] > maxPathReg) {
        maxPathReg = stHdl->pitchMaxPathReg[1][idx];
        tmpInt = idx;
      }
    }
    stHdl->pitchMaxPathAll = maxPathReg;
    stHdl->bestPeriodEst = tmpInt;

    memcpy(&(stHdl->pitchMaxPathReg[0][0]), &(stHdl->pitchMaxPathReg[1][0]),
           sizeof(float) * stHdl->maxPeriod);
    for (idx = 0; idx < stHdl->difPeriod; idx++) {
      stHdl->pitchMaxPathReg[0][idx] -= maxPathReg;
    }
  }

  tmpInt = stHdl->bestPeriodEst;
  frmCorr = 0;
  // Backward pass
  for (sub = (stHdl->nFeat * 2) - 1; sub >= 0; sub--) {
    bestPeriodEstLocal[sub] = stHdl->maxPeriod - tmpInt;

    XCIdx = sub + (stHdl->xCorrOffsetIdx * 2);
    if (XCIdx >= (2 * stHdl->nFeat)) {
      XCIdx -= (2 * stHdl->nFeat);
    }
    frmCorr += stHdl->frmWeightNorm[sub] * stHdl->xCorrTmp[XCIdx][tmpInt];
    tmpInt = stHdl->pitchPrev[sub][tmpInt];
  }
  frmCorr = AUP_PE_MAX(0, frmCorr / (float)(stHdl->nFeat * 2));
  stHdl->voiced = (frmCorr >= stHdl->dynamCfg.voicedThr) ? 1 : 0;

  for (sub = 0; sub < (stHdl->nFeat * 2); sub++) {
    w = stHdl->frmWeightNorm[sub];
    sw += w;
    sx += w * sub;
    sxx += w * sub * sub;
    sxy += w * sub * bestPeriodEstLocal[sub];
    sy += w * bestPeriodEstLocal[sub];
  }

  // Linear regression to figure out the pitch contour
  // frmCorrCorrection = frmCorr;
  tmpDenom = (sw * sxx - sx * sx);
  if (tmpDenom == 0)
    bestA = (sw * sxy - sx * sy) / 1e-15f;
  else
    bestA = (sw * sxy - sx * sy) / tmpDenom;

  if (stHdl->voiced == 1) {
    tmpDenom = (sy / sw) / (4 * 2 * stHdl->nFeat);
    bestA = AUP_PE_MIN(tmpDenom, AUP_PE_MAX(-tmpDenom, bestA));
  } else {  // if there is no voice inside this frame
    bestA = 0;
  }
  bestB = (sy - bestA * sx) / sw;
  estimatedPeriod = bestB + 5.5f * bestA;

  if (stHdl->voiced == 1) {
    stHdl->pitchEstResult =
        ((float)(stHdl->stCfg.procFs)) / AUP_PE_MAX(1.0f, estimatedPeriod);
  } else {
    stHdl->pitchEstResult = 0;
  }

  if (pOut != NULL) {
    pOut->pitchFreq = stHdl->pitchEstResult;
    pOut->voiced = stHdl->voiced;
  }

  return 0;
}
