//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#include "biquad.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "biquad_st.h"

#define AUP_BIQUAD_NUM_DUMP_FILES (20)
#define AUP_BIQUAD_DUMP_FILENAMES (200)

// ==========================================================================================
// internal tools
// ==========================================================================================

static int AUP_Biquad_checkStatCfg(const Biquad_StaticCfg* pCfg) {
  int secIdx;
  if (pCfg == NULL) {
    return -1;
  }

  if (pCfg->maxNSample == 0 ||
      pCfg->maxNSample > AGORA_UAP_BIQUAD_MAX_INPUT_LEN) {
    return -1;
  }
  if (pCfg->nsect > AGORA_UAP_BIQUAD_MAX_SECTION) {
    return -1;
  }

  // if external filter coefficients are required, we need to check the
  //    external filter coeff pointers' validness
  if (pCfg->nsect > 0) {
    for (secIdx = 0; secIdx < pCfg->nsect; secIdx++) {
      if (pCfg->B[secIdx] == NULL || pCfg->A[secIdx] == NULL) {
        return -1;
      }
    }
    if (pCfg->G == NULL) {
      return -1;
    }
  }

  return 0;
}

static int AUP_Biquad_publishStaticCfg(Biquad_St* stHdl) {
  const Biquad_StaticCfg* pStatCfg;
  int idx;

  if (stHdl == NULL) {
    return -1;
  }
  pStatCfg = (const Biquad_StaticCfg*)(&(stHdl->stCfg));

  stHdl->maxNSample = (int)pStatCfg->maxNSample;

  // first, give default (all-pass-filter) values to filter coeffs
  for (idx = 0; idx < AGORA_UAP_BIQUAD_MAX_SECTION; idx++) {
    stHdl->BCoeff[idx][0] = 1.0f;
    stHdl->BCoeff[idx][1] = 0;
    stHdl->BCoeff[idx][2] = 0;
    stHdl->ACoeff[idx][0] = 1.0f;
    stHdl->ACoeff[idx][1] = 0;
    stHdl->ACoeff[idx][2] = 0;
    stHdl->GCoeff[idx] = 1.0f;
  }

  if (pStatCfg->nsect <= 0) {
    stHdl->nsect = _BIQUAD_DC_REMOVAL_NSECT;
    for (idx = 0; idx < stHdl->nsect; idx++) {
      stHdl->BCoeff[idx][0] = _BIQUAD_DC_REMOVAL_B[idx][0];
      stHdl->BCoeff[idx][1] = _BIQUAD_DC_REMOVAL_B[idx][1];
      stHdl->BCoeff[idx][2] = _BIQUAD_DC_REMOVAL_B[idx][2];
      stHdl->ACoeff[idx][0] = _BIQUAD_DC_REMOVAL_A[idx][0];
      stHdl->ACoeff[idx][1] = _BIQUAD_DC_REMOVAL_A[idx][1];
      stHdl->ACoeff[idx][2] = _BIQUAD_DC_REMOVAL_A[idx][2];
      stHdl->GCoeff[idx] = _BIQUAD_DC_REMOVAL_G[idx];
    }
  } else {
    stHdl->nsect = pStatCfg->nsect;
    for (idx = 0; idx < stHdl->nsect; idx++) {
      stHdl->BCoeff[idx][0] = pStatCfg->B[idx][0];
      stHdl->BCoeff[idx][1] = pStatCfg->B[idx][1];
      stHdl->BCoeff[idx][2] = pStatCfg->B[idx][2];

      stHdl->ACoeff[idx][0] = pStatCfg->A[idx][0];
      stHdl->ACoeff[idx][1] = pStatCfg->A[idx][1];
      stHdl->ACoeff[idx][2] = pStatCfg->A[idx][2];

      stHdl->GCoeff[idx] = pStatCfg->G[idx];
    }
  }

  return 0;
}

static int AUP_Biquad_resetVariables(Biquad_St* stHdl) {
  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);
  memset(stHdl->sectW, 0, sizeof(stHdl->sectW));

  return 0;
}

// ==========================================================================================
// public APIS
// ==========================================================================================

int AUP_Biquad_create(void** stPtr) {
  Biquad_St* tmpPtr;

  if (stPtr == NULL) {
    return -1;
  }
  *stPtr = (void*)malloc(sizeof(Biquad_St));
  if (*stPtr == NULL) {
    return -1;
  }
  memset(*stPtr, 0, sizeof(Biquad_St));

  tmpPtr = (Biquad_St*)(*stPtr);

  tmpPtr->dynamMemPtr = NULL;
  tmpPtr->dynamMemSize = 0;

  tmpPtr->stCfg.maxNSample = 768;
  tmpPtr->stCfg.nsect = 0;
  for (int idx = 0; idx < AGORA_UAP_BIQUAD_MAX_SECTION; idx++) {
    tmpPtr->stCfg.A[idx] = NULL;
    tmpPtr->stCfg.B[idx] = NULL;
  }
  tmpPtr->stCfg.G = NULL;

  return 0;
}

int AUP_Biquad_destroy(void** stPtr) {
  Biquad_St* stHdl;

  if (stPtr == NULL) {
    return 0;
  }

  stHdl = (Biquad_St*)(*stPtr);
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

int AUP_Biquad_memAllocate(void* stPtr, const Biquad_StaticCfg* pCfg) {
  Biquad_St* stHdl = NULL;
  char* memPtr = NULL;
  int maxNSample, nsect, idx;

  int inputTempBufMemSize = 0;
  int sectOutputBufMemSize_EACH = 0;
  int totalMemSize = 0;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (Biquad_St*)(stPtr);

  if (AUP_Biquad_checkStatCfg(pCfg) < 0) {
    return -1;
  }
  memcpy(&(stHdl->stCfg), pCfg, sizeof(Biquad_StaticCfg));

  if (AUP_Biquad_publishStaticCfg(stHdl) < 0) {
    return -1;
  }
  maxNSample = stHdl->maxNSample;
  nsect = stHdl->nsect;

  // check memory requirement
  inputTempBufMemSize = AGORA_UAP_BIQUAD_ALIGN8(sizeof(float) * maxNSample);
  totalMemSize += inputTempBufMemSize;

  sectOutputBufMemSize_EACH =
      AGORA_UAP_BIQUAD_ALIGN8(sizeof(float) * maxNSample);
  totalMemSize += sectOutputBufMemSize_EACH * nsect;

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
  memPtr = (char*)(stHdl->dynamMemPtr);

  stHdl->inputTempBuf = (float*)memPtr;
  memPtr += inputTempBufMemSize;

  for (idx = 0; idx < nsect; idx++) {
    stHdl->sectOutputBuf[idx] = (float*)memPtr;
    memPtr += sectOutputBufMemSize_EACH;
  }
  for (; idx < AGORA_UAP_BIQUAD_MAX_SECTION; idx++) {
    stHdl->sectOutputBuf[idx] = NULL;
  }

  if (((int)(memPtr - (char*)(stHdl->dynamMemPtr))) > totalMemSize) {
    return -1;
  }

  return 0;
}

int AUP_Biquad_init(void* stPtr) {
  Biquad_St* stHdl;

  if (stPtr == NULL) {
    return -1;
  }
  stHdl = (Biquad_St*)(stPtr);

  if (AUP_Biquad_resetVariables(stHdl) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Biquad_getStaticCfg(const void* stPtr, Biquad_StaticCfg* pCfg) {
  const Biquad_St* stHdl;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (const Biquad_St*)(stPtr);

  memcpy(pCfg, &(stHdl->stCfg), sizeof(Biquad_StaticCfg));

  return 0;
}

int AUP_Biquad_getAlgDelay(const void* stPtr, int* delayInSamples) {
  const Biquad_St* stHdl;

  if (stPtr == NULL || delayInSamples == NULL) {
    return -1;
  }
  stHdl = (const Biquad_St*)(stPtr);

  *delayInSamples = stHdl->nsect;

  return 0;
}

int AUP_Biquad_proc(void* stPtr, const Biquad_InputData* pIn,
                    Biquad_OutputData* pOut) {
  Biquad_St* stHdl = NULL;
  int isFloatIO = 0;
  int inputNSamples, nSect;
  int sectIdx, smplIdx;
  float tmp1;
  const short* pShortTemp;
  float* src;
  float* tgt;

  if (stPtr == NULL || pIn == NULL || pOut == NULL) {  //  pCtrl == NULL
    return -1;
  }
  if (pIn->samplesPtr == NULL || pOut->outputBuff == NULL) {
    return -1;
  }

  stHdl = (Biquad_St*)(stPtr);

  if (((int)pIn->nsamples) > stHdl->maxNSample) {
    return -1;
  }

  isFloatIO = 0;
  if (pIn->sampleType != 0) {
    isFloatIO = 1;
  }

  inputNSamples = (int)pIn->nsamples;
  nSect = stHdl->nsect;

  // special handle for input
  if (isFloatIO == 0) {
    pShortTemp = (const short*)pIn->samplesPtr;
    for (smplIdx = 0; smplIdx < inputNSamples; smplIdx++) {
      stHdl->inputTempBuf[smplIdx] = (float)pShortTemp[smplIdx];
    }
  } else {
    memcpy(stHdl->inputTempBuf, (const float*)pIn->samplesPtr,
           sizeof(float) * inputNSamples);
  }

  for (sectIdx = 0; sectIdx < nSect; sectIdx++) {
    if (sectIdx == 0) {
      src = stHdl->inputTempBuf;
    } else {
      src = stHdl->sectOutputBuf[sectIdx - 1];
    }
    tgt = stHdl->sectOutputBuf[sectIdx];

    for (smplIdx = 0; smplIdx < inputNSamples; smplIdx++) {
      tmp1 = src[smplIdx] -
             stHdl->ACoeff[sectIdx][1] * stHdl->sectW[sectIdx][0] -
             stHdl->ACoeff[sectIdx][2] * stHdl->sectW[sectIdx][1];

      tgt[smplIdx] = stHdl->GCoeff[sectIdx] *
                     (stHdl->BCoeff[sectIdx][0] * tmp1 +
                      stHdl->BCoeff[sectIdx][1] * stHdl->sectW[sectIdx][0] +
                      stHdl->BCoeff[sectIdx][2] * stHdl->sectW[sectIdx][1]);

      stHdl->sectW[sectIdx][1] = stHdl->sectW[sectIdx][0];
      stHdl->sectW[sectIdx][0] = tmp1;
    }
  }

  // prepare output buffer
  if (isFloatIO == 0) {
    for (smplIdx = 0; smplIdx < inputNSamples; smplIdx++) {
      ((short*)pOut->outputBuff)[smplIdx] =
          (short)_BIQUAD_FLOAT2SHORT(stHdl->sectOutputBuf[nSect - 1][smplIdx]);
    }
  } else {
    memcpy(pOut->outputBuff, stHdl->sectOutputBuf[nSect - 1],
           sizeof(float) * inputNSamples);
  }

  return 0;
}
