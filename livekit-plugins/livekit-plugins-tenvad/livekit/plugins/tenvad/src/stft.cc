//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stft.h"
#include "stft_st.h"
#include "fftw.h"

// ==========================================================================================
// internal tools
// ==========================================================================================

static int AUP_Analyzer_checkStatCfg(Analyzer_StaticCfg* pCfg) {
  if (pCfg == NULL) {
    return -1;
  }

  if (pCfg->fft_size != 256 && pCfg->fft_size != 512 &&
      pCfg->fft_size != 1024 && pCfg->fft_size != 2048 &&
      pCfg->fft_size != 4096) {
    return -1;
  }

  if (pCfg->win_len <= 0 || pCfg->win_len < pCfg->hop_size ||
      pCfg->win_len > pCfg->fft_size) {
    return -1;
  }

  if (pCfg->hop_size <= 0) {
    return -1;
  }

  return 0;
}

static int AUP_Analyzer_publishStaticCfg(Analyzer_St* stHdl) {
  const Analyzer_StaticCfg* pStatCfg;
  int idx;

  if (stHdl == NULL) {
    return -1;
  }
  pStatCfg = (const Analyzer_StaticCfg*)(&(stHdl->stCfg));

  stHdl->nBins = (pStatCfg->fft_size >> 1) + 1;
  if (pStatCfg->ana_win_coeff != NULL) {
    memcpy(stHdl->windowCoffCopy, pStatCfg->ana_win_coeff,
           sizeof(float) * pStatCfg->win_len);
  } else {
    for (idx = 0; idx < AUP_STFT_MAX_FFTSZ; idx++) {
      stHdl->windowCoffCopy[idx] = 1.0f;
    }
  }
  return 0;
}

static int AUP_Analyzer_resetVariables(Analyzer_St* stHdl) {
  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);
  return 0;
}

static int AUP_Analyzer_dynamMemPrepare(Analyzer_St* stHdl, void* memPtrExt,
                                        size_t memSize) {
  int inputQMemSz = 0;
  int fftInputBufMemSz = 0;
  int totalMemSize = 0;
  char* memPtr = NULL;

  inputQMemSz = AUP_STFT_ALIGN8(sizeof(float) * (stHdl->stCfg.win_len + 4));
  totalMemSize += inputQMemSz;

  fftInputBufMemSz =
      AUP_STFT_ALIGN8(sizeof(float) * (stHdl->stCfg.fft_size + 4));
  totalMemSize += fftInputBufMemSz;

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

  stHdl->inputQ = (float*)memPtr;
  memPtr += inputQMemSz;

  stHdl->fftInputBuf = (float*)memPtr;
  memPtr += fftInputBufMemSz;

  if (((int)(memPtr - (char*)memPtrExt)) > totalMemSize) {
    return -1;
  }

  return (totalMemSize);
}

// ==========================================================================================
// public APIs
// ==========================================================================================

int AUP_Analyzer_create(void** stPtr) {
  Analyzer_St* tmpPtr;

  if (stPtr == NULL) {
    return -1;
  }

  *stPtr = (void*)malloc(sizeof(Analyzer_St));
  if (*stPtr == NULL) {
    return -1;
  }
  memset(*stPtr, 0, sizeof(Analyzer_St));

  tmpPtr = (Analyzer_St*)(*stPtr);

  tmpPtr->dynamMemPtr = NULL;
  tmpPtr->dynamMemSize = 0;

  tmpPtr->stCfg.win_len = 768;
  tmpPtr->stCfg.hop_size = 256;
  tmpPtr->stCfg.fft_size = 1024;
  tmpPtr->stCfg.ana_win_coeff = NULL;

  return 0;
}

int AUP_Analyzer_destroy(void** stPtr) {
  Analyzer_St* stHdl;

  if (stPtr == NULL) {
    return 0;
  }

  stHdl = (Analyzer_St*)(*stPtr);
  if (stHdl == NULL) {
    return 0;
  }

  if (stHdl->dynamMemPtr != NULL) {
    free(stHdl->dynamMemPtr);
  }
  stHdl->dynamMemPtr = NULL;

  free(stHdl);
  (*stPtr) = NULL;

  return 0;
}

int AUP_Analyzer_memAllocate(void* stPtr, const Analyzer_StaticCfg* pCfg) {
  Analyzer_St* stHdl = NULL;
  Analyzer_StaticCfg localStCfg;
  int totalMemSize = 0;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (Analyzer_St*)(stPtr);

  memcpy(&localStCfg, pCfg, sizeof(Analyzer_StaticCfg));
  if (AUP_Analyzer_checkStatCfg(&localStCfg) < 0) {
    return -1;
  }

  memcpy(&(stHdl->stCfg), &localStCfg, sizeof(Analyzer_StaticCfg));

  // 1st. publish internal static configuration registers
  if (AUP_Analyzer_publishStaticCfg(stHdl) < 0) {
    return -1;
  }

  // 4th: check memory requirement
  totalMemSize = AUP_Analyzer_dynamMemPrepare(stHdl, NULL, 0);
  if (totalMemSize < 0) {
    return -1;
  }

  // 5th: allocate dynamic memory
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

  // 6th: setup the pointers/variable
  if (AUP_Analyzer_dynamMemPrepare(stHdl, stHdl->dynamMemPtr,
                                   stHdl->dynamMemSize) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Analyzer_init(void* stPtr) {
  Analyzer_St* stHdl;

  if (stPtr == NULL) {
    return -1;
  }
  stHdl = (Analyzer_St*)(stPtr);

  if (AUP_Analyzer_resetVariables(stHdl) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Analyzer_getStaticCfg(const void* stPtr, Analyzer_StaticCfg* pCfg) {
  const Analyzer_St* stHdl;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (const Analyzer_St*)(stPtr);

  memcpy(pCfg, &(stHdl->stCfg), sizeof(Analyzer_StaticCfg));

  return 0;
}

int AUP_Analyzer_proc(void* stPtr, const Analyzer_InputData* pIn,
                      Analyzer_OutputData* pOut) {
  Analyzer_St* stHdl = NULL;
  int hopSz, fftSz, winLen, nBins;
  int idx = 0;

  if (stPtr == NULL || pIn == NULL || pIn->input == NULL || pOut == NULL ||
      pOut->output == NULL) {
    return -1;
  }
  stHdl = (Analyzer_St*)(stPtr);

  if (pIn->iLength != stHdl->stCfg.hop_size ||
      pOut->oLength < stHdl->stCfg.fft_size) {
    return -1;
  }
  hopSz = stHdl->stCfg.hop_size;
  fftSz = stHdl->stCfg.fft_size;
  nBins = (fftSz >> 1) + 1;
  winLen = stHdl->stCfg.win_len;

  memset(pOut->output, 0, sizeof(float) * pOut->oLength);
  memmove(stHdl->inputQ, stHdl->inputQ + hopSz,
          sizeof(float) * (winLen - hopSz));
  memcpy(stHdl->inputQ + (winLen - hopSz), pIn->input, sizeof(float) * hopSz);

  if (stHdl->stCfg.ana_win_coeff != NULL) {
    for (idx = 0; idx < winLen; idx++) {
      stHdl->fftInputBuf[idx] = stHdl->inputQ[idx] * stHdl->windowCoffCopy[idx];
    }
  } else {
    for (idx = 0; idx < winLen; idx++) {
      stHdl->fftInputBuf[idx] = stHdl->inputQ[idx];
    }
  }
  for (; idx < fftSz; idx++) {
    stHdl->fftInputBuf[idx] = 0;
  }

  if (fftSz == 256) {
    AUP_FFTW_r2c_256(stHdl->fftInputBuf, pOut->output);
  } else if (fftSz == 512) {
    AUP_FFTW_r2c_512(stHdl->fftInputBuf, pOut->output);
  } else if (fftSz == 1024) {
    AUP_FFTW_r2c_1024(stHdl->fftInputBuf, pOut->output);
  } else if (fftSz == 2048) {
    AUP_FFTW_r2c_2048(stHdl->fftInputBuf, pOut->output);
  } else if (fftSz == 4096) {
    AUP_FFTW_r2c_4096(stHdl->fftInputBuf, pOut->output);
  }
  AUP_FFTW_InplaceTransf(1, fftSz, pOut->output);
  AUP_FFTW_RescaleFFTOut(fftSz, pOut->output);

  return 0;
}
