//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __FSCVRT_ST_H__
#define __FSCVRT_ST_H__

#include <stdio.h>

#include "fscvrt.h"

#define _FSCVRT_MAXNSEC (40)
#define _FSCVRT_COMMON_FS (96000)
#define _FSCVRT_ALIGN8(o) (((o) + 7) & (~7))
#define _FSCVRT_FLOAT2SHORT(x) \
  ((x) < -32767.5f ? -32768 : ((x) > 32766.5f ? 32767 : (short)floor(.5 + (x))))
#define _FSCVRT_MIN(x, y) ((x > y) ? (y) : (x))
#define _FSCVRT_MAX(x, y) ((x > y) ? (x) : (y))

#define _FSCVRT_1over2_LOWPASS_NSEC (5)
static const float _FSCVRT_1over2_LOWPASS_B[_FSCVRT_1over2_LOWPASS_NSEC][3] = {
    {1.000000e+00f, 1.830863e+00f, 1.000000e+00f},
    {1.000000e+00f, 1.039654e+00f, 1.000000e+00f},
    {1.000000e+00f, 4.900788e-01f, 1.000000e+00f},
    {1.000000e+00f, 2.419292e-01f, 1.000000e+00f},
    {1.000000e+00f, 1.517919e-01f, 1.000000e+00f}};
static const float _FSCVRT_1over2_LOWPASS_A[_FSCVRT_1over2_LOWPASS_NSEC][3] = {
    {1.000000e+00f, -8.445478e-01f, 2.453003e-01f},
    {1.000000e+00f, -5.469711e-01f, 5.010509e-01f},
    {1.000000e+00f, -2.646897e-01f, 7.464574e-01f},
    {1.000000e+00f, -1.074159e-01f, 8.912371e-01f},
    {1.000000e+00f, -4.448528e-02f, 9.702184e-01f}};
static const float _FSCVRT_1over2_LOWPASS_G[_FSCVRT_1over2_LOWPASS_NSEC] = {
    // 4.184914e-01f,4.184914e-01f,4.184914e-01f,4.184914e-01f,4.184914e-01f
    4.233410e-01f, 4.233410e-01f, 4.233410e-01f, 4.233410e-01f, 4.233410e-01f};

#define _FSCVRT_1over3_LOWPASS_NSEC (5)
static const float _FSCVRT_1over3_LOWPASS_B[_FSCVRT_1over3_LOWPASS_NSEC][3] = {
    {1.000000e+00f, 1.535971e+00f, 1.000000e+00f},
    {1.000000e+00f, 6.284728e-02f, 1.000000e+00f},
    {1.000000e+00f, -5.726159e-01f, 1.000000e+00f},
    {1.000000e+00f, -7.990919e-01f, 1.000000e+00f},
    {1.000000e+00f, -8.741772e-01f, 1.000000e+00f}};
static const float _FSCVRT_1over3_LOWPASS_A[_FSCVRT_1over3_LOWPASS_NSEC][3] = {
    {1.000000e+00f, -1.261229e+00f, 4.351921e-01f},
    {1.000000e+00f, -1.171732e+00f, 6.072938e-01f},
    {1.000000e+00f, -1.078980e+00f, 7.901941e-01f},
    {1.000000e+00f, -1.026436e+00f, 9.073955e-01f},
    {1.000000e+00f, -1.013524e+00f, 9.743813e-01f}};
static const float _FSCVRT_1over3_LOWPASS_G[_FSCVRT_1over3_LOWPASS_NSEC] = {
    // 3.126979e-01f,3.126979e-01f,3.126979e-01f,3.126979e-01f,3.126979e-01f
    3.1704682e-01f, 3.1704682e-01f, 3.1704682e-01f, 3.1704682e-01f,
    3.1704682e-01f};

#define _FSCVRT_1over4_LOWPASS_NSEC (5)
static const float _FSCVRT_1over4_LOWPASS_B[_FSCVRT_1over4_LOWPASS_NSEC][3] = {
    {1.000000e+00f, 1.193034e+00f, 1.000000e+00f},
    {1.000000e+00f, -5.757392e-01f, 1.000000e+00f},
    {1.000000e+00f, -1.105338e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.271233e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.323929e+00f, 1.000000e+00f}};
static const float _FSCVRT_1over4_LOWPASS_A[_FSCVRT_1over4_LOWPASS_NSEC][3] = {
    {1.000000e+00f, -1.447526e+00f, 5.478735e-01f},
    {1.000000e+00f, -1.429707e+00f, 6.830356e-01f},
    {1.000000e+00f, -1.412017e+00f, 8.292100e-01f},
    {1.000000e+00f, -1.405145e+00f, 9.242718e-01f},
    {1.000000e+00f, -1.412679e+00f, 9.790443e-01f}};
static const float _FSCVRT_1over4_LOWPASS_G[_FSCVRT_1over4_LOWPASS_NSEC] = {
    // 2.700060e-01f,2.700060e-01f,2.700060e-01f,2.700060e-01f,2.700060e-01f
    2.7502688e-01f, 2.7502688e-01f, 2.7502688e-01f, 2.7502688e-01f,
    2.7502688e-01f};

#define _FSCVRT_1over6_LOWPASS_NSEC (5)
static const float _FSCVRT_1over6_LOWPASS_B[_FSCVRT_1over6_LOWPASS_NSEC][3] = {
    {1.000000e+00f, 4.149228e-01f, 1.000000e+00f},
    {1.000000e+00f, -1.285358e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.583012e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.663823e+00f, 1.000000e+00f},
    {1.000000e+00f, -1.688104e+00f, 1.000000e+00f}};
static const float _FSCVRT_1over6_LOWPASS_A[_FSCVRT_1over6_LOWPASS_NSEC][3] = {
    {1.000000e+00f, -1.688731e+00f, 7.264798e-01f},
    {1.000000e+00f, -1.696982e+00f, 8.146896e-01f},
    {1.000000e+00f, -1.706117e+00f, 9.049889e-01f},
    {1.000000e+00f, -1.713737e+00f, 9.598250e-01f},
    {1.000000e+00f, -1.723161e+00f, 9.892408e-01f}};
static const float _FSCVRT_1over6_LOWPASS_G[_FSCVRT_1over6_LOWPASS_NSEC] = {
    // 2.333130e-01f,2.333130e-01f,2.333130e-01f,2.333130e-01f,2.333130e-01f
    2.3765156e-01f, 2.3765156e-01f, 2.3765156e-01f, 2.3765156e-01f,
    2.3765156e-01f};

typedef struct FscvrtSt_ {
  void* dynamMemPtr;    // memory pointer holding the dynamic memory
  size_t dynamMemSize;  // size of the buffer *dynamMemPtr

  // Static Configuration
  FscvrtStaticCfg stCfg;

  // Internal Static Config Registers, which are generated from stCfg
  int mode;
  // 0: direct bypass, 1: direct upsampling, 2: direct downsampling
  // 3: upsampling->downsampling
  int upSmplRate;
  int downSmplRate;
  int biquadInBufLen;   // biquad input buffer length
  int biquadOutBufLen;  // biquad output buffer length

  int nSec;
  const float* biquadB[_FSCVRT_MAXNSEC];
  const float* biquadA[_FSCVRT_MAXNSEC];
  const float* biquadG;  // gain for each section

  // ---------------------------------------------------------------
  // Variables
  void* biquadSt;  // biqua filter state handler
  int biquadInBufCnt;
  float* biquadInBuf;  // [biquadInBufLen]
  int biquadOutBufCnt;
  float* biquadOutBuf;  // [biquadOutBufLen]
} FscvrtSt;

#endif  // __FSCVRT_ST_H__
