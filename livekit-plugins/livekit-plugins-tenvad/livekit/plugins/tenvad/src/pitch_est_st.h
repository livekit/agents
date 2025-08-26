//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __PITCH_EST_ST_H__
#define __PITCH_EST_ST_H__

#include <stdio.h>

#include "pitch_est.h"

#define AUP_PE_ALIGN8(o) (((o) + 7) & (~7))
#define AUP_PE_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define AUP_PE_MIN(x, y) (((x) > (y)) ? (y) : (x))

#define AUP_PE_NB_BANDS (18)  // number of mel bands
#define AUP_PE_LPC_ORDER (16)

#define AUP_PE_XCORR_TRAINING_OFFSET (80)  // = 80
#define AUP_PE_MIN_PERIOD_16KHZ (32)   // stands for 1.333kHz, PITCH_MIN_PERIOD
#define AUP_PE_MAX_PERIOD_16KHZ (256)  // stands for ~62Hz, PITCH_MAX_PERIOD

#define AUP_PE_LOWPSS_NSEC (5)
const float AUP_PE_B_2KHZ[AUP_PE_LOWPSS_NSEC][3] = {
    {1.000000e+00f, -1.303155e-01f, 1.000000e+00f},
    {1.000000e+00f, -1.563002e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.759739e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.811650e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.827332e+00f, 1.000000e+00f}};
const float AUP_PE_A_2KHZ[AUP_PE_LOWPSS_NSEC][3] = {
    {1.000000e+00f, -1.726800e+00f, 7.526543e-01f},
    {1.000000e+00f, -1.762977e+00f, 8.277960e-01f},
    {1.000000e+00f, -1.802014e+00f, 9.079320e-01f},
    {1.000000e+00f, -1.828423e+00f, 9.594240e-01f},
    {1.000000e+00f, -1.846774e+00f, 9.888390e-01f}};
const float AUP_PE_G_2KHZ[AUP_PE_LOWPSS_NSEC] = {
    2.156619e-01f, 2.156619e-01f, 2.156619e-01f, 2.156619e-01f, 2.156619e-01f};

const float AUP_PE_B_4KHZ[AUP_PE_LOWPSS_NSEC][3] = {
    {1.000000e+00f, 1.198825e+00f, 1.000000e+00f},
    {1.000000e+00f, -5.674614e-01f, 1.000000e+00f},
    {1.000000e+00f, -1.099061e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.265846e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.318849e+00f, 1.000000e+00f}};
const float AUP_PE_A_4KHZ[AUP_PE_LOWPSS_NSEC][3] = {
    {1.000000e+00f, -1.445267e+00f, 5.463974e-01f},
    {1.000000e+00f, -1.426720e+00f, 6.820138e-01f},
    {1.000000e+00f, -1.408255e+00f, 8.286664e-01f},
    {1.000000e+00f, -1.400909e+00f, 9.240320e-01f},
    {1.000000e+00f, -1.408242e+00f, 9.789776e-01f}};
const float AUP_PE_G_4KHZ[AUP_PE_LOWPSS_NSEC] = {
    2.692541e-01f, 2.692541e-01f, 2.692541e-01f, 2.692541e-01f, 2.692541e-01f};

const float AUP_PE_B_8KHZ[AUP_PE_LOWPSS_NSEC][3] = {
    {1.000000e+00f, 1.830863e+00f, 1.000000e+00f},
    {1.000000e+00f, 1.039654e+00f, 1.000000e+00f},
    {1.000000e+00f, 4.900788e-01f, 1.000000e+00f},
    {1.000000e+00f, 2.419292e-01f, 1.000000e+00f},
    {1.000000e+00f, 1.517919e-01f, 1.000000e+00f}};
const float AUP_PE_A_8KHZ[AUP_PE_LOWPSS_NSEC][3] = {
    {1.000000e+00f, -8.445478e-01f, 2.453003e-01f},
    {1.000000e+00f, -5.469711e-01f, 5.010509e-01f},
    {1.000000e+00f, -2.646897e-01f, 7.464574e-01f},
    {1.000000e+00f, -1.074159e-01f, 8.912371e-01f},
    {1.000000e+00f, -4.448528e-02f, 9.702184e-01f}};
const float AUP_PE_G_8KHZ[AUP_PE_LOWPSS_NSEC] = {
    4.165686e-01f, 4.165686e-01f, 4.165686e-01f, 4.165686e-01f, 4.165686e-01f};

#define AUP_PE_PI (3.1415926f)

#define AUP_PE_FEAT_TIME_WINDOW (40)  // in ms
// how much time data to use for cross-correlation calculation
#define AUP_PE_FEAT_MAX_NFRM (12)
#define AUP_PE_TOTAL_NFEAT (55)

#define AUP_PE_ASSUMED_FFT_4_BAND_ENG (80)
const int AUP_PE_BAND_START_INDEX[AUP_PE_NB_BANDS] = {
    // 0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40};
// compensation
const float AUP_PE_BAND_LPC_COMP[AUP_PE_NB_BANDS] = {
    0.8f, 1.f,  1.f,  1.f,       1.f,   1.f,   1.f,  1.f,       0.666667f,
    0.5f, 0.5f, 0.5f, 0.333333f, 0.25f, 0.25f, 0.2f, 0.166667f, 0.173913f};

#define AUP_PE_PITCHMAXPATH_W (0.02f)

typedef struct PE_St_ {
  void* dynamMemPtr;    // memory pointer holding the dynamic memory
  size_t dynamMemSize;  // size of the buffer *dynamMemPtr
  // void* ifftStHdl;  // AgoraFft*
  void* biquadIIRPtr;

  // ---------------------------------------------------------------
  // Static Configuration
  PE_StaticCfg stCfg;

  // ---------------------------------------------------------------
  // Internal Static Config Registers, which are generated from stCfg
  int nBins;
  int procResampleRate;     // AUP_PE_FS / stCfg.procFs
  int minPeriod;            // min. pitch period in procFs
  int maxPeriod;            // max. pitch period in procFs
  int difPeriod;            // maxPeriod - minPeriod
  int inputResampleBufLen;  // length of the resampling output buffer
  int inputQLen;            // MAX(TRAINING_OFFSET_XXXX, hopSz) + hopSz
  int excBufLen;            // PITCH_MAX_PERIOD + hopSz + 1
  int nFeat;                // number of feature frames to use/store
  int estDelay;             // pitch estimate delay in terms of frames
  float dct_table[AUP_PE_NB_BANDS *
                  AUP_PE_NB_BANDS];  // coeff. table of DCT transformation

  // ---------------------------------------------------------------
  // Dynamic Configuration
  PE_DynamCfg dynamCfg;

  // ---------------------------------------------------------------
  // Internal Dynamic Config Registers, which are generated from dynamCfg

  // ---------------------------------------------------------------
  // Variables
  /////////////////////////////////////////////////////////////////////////
  float* inputResampleBuf;  // [inputResampleBufLen]
  int inputResampleBufIdx;

  float* inputQ;           // [inputQLen]
  float* alignedIn;        // [hopSz]
  float* lpcFilterOutBuf;  // [hopSz]

  float* excBuf;  // [excBufLen]
  // excBuf stores the smoothed LPC prediction result
  float* excBufSq;  // [excBufLen]
  // = excBuf.^2

  float lpc[AUP_PE_LPC_ORDER];
  float pitch_mem[AUP_PE_LPC_ORDER];
  float pitch_filt;

  float tmpFeat[AUP_PE_TOTAL_NFEAT];

  int xCorrOffsetIdx;  // the oldest frame's index in xCorr Buffer
  float* xCorrInst;    // [maxPeriod]
  float* xCorr[AUP_PE_FEAT_MAX_NFRM * 2];  // [stHdl->nFeat * 2][maxPeriod + 1]
  // circ-buffer [<--->][...]
  float*
      xCorrTmp[AUP_PE_FEAT_MAX_NFRM * 2];  // [stHdl->nFeat * 2][maxPeriod + 1]
  // temporarily modified version of xCorr

  float frmWeight[AUP_PE_FEAT_MAX_NFRM * 2];
  float frmWeightNorm[AUP_PE_FEAT_MAX_NFRM * 2];
  // normalized version of frmWeight

  // variables for best pitch estimation ....
  /////////////////////////////////////////////////////////////////////////
  float* pitchMaxPathReg[2];  // [2][maxPeriod]

  int* pitchPrev[AUP_PE_FEAT_MAX_NFRM * 2];  // [stHdl->nFeat * 2][maxPeriod]
  float pitchMaxPathAll;
  int bestPeriodEst;

  int voiced;            // whether this frame has voice
  float pitchEstResult;  // the final result storing the estimated pitch freq.
} PE_St;

#endif  // __PITCH_EST_ST_H__
