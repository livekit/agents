//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __STFT_ST_H__
#define __STFT_ST_H__

#include <stdio.h>
#include "stft.h"

#define AUP_STFT_ALIGN8(o) (((o) + 7) & (~7))
#define AUP_STFT_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define AUP_STFT_MIN(x, y) (((x) > (y)) ? (y) : (x))

typedef struct Analyzer_St_ {
  void* dynamMemPtr;    // memory pointer holding the dynamic memory
  size_t dynamMemSize;  // size of the buffer *dynamMemPtr

  // ---------------------------------------------------------------
  // Static Configuration
  Analyzer_StaticCfg stCfg;

  // ---------------------------------------------------------------
  // Internal Static Config Registers, which are generated from stCfg
  int nBins;
  float windowCoffCopy[AUP_STFT_MAX_FFTSZ];

  // ---------------------------------------------------------------
  // Dynamic Configuration

  // ---------------------------------------------------------------
  // Internal Dynamic Config Registers, which are generated from dynamCfg

  // ---------------------------------------------------------------
  // Variables
  float* inputQ;       // [stCfg->win_len + 4]
  float* fftInputBuf;  // [stCfg->fft_size + 4]
} Analyzer_St;

#endif  // __STFT_ST_H__
