//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "fscvrt.h"
#include "fscvrt_st.h"
#include "biquad.h"

// ==========================================================================================
// internal tools
// ==========================================================================================

static int AUP_Fscvrt_FilterSet(int resampleRate, int* nsect,
                                const float* B[_FSCVRT_MAXNSEC],
                                const float* A[_FSCVRT_MAXNSEC],
                                const float** G) {
  int idx;
  if (resampleRate == 2) {
    *nsect = _FSCVRT_1over2_LOWPASS_NSEC;
    for (idx = 0; idx < (*nsect); idx++) {
      B[idx] = &(_FSCVRT_1over2_LOWPASS_B[idx][0]);
      A[idx] = &(_FSCVRT_1over2_LOWPASS_A[idx][0]);
    }
    *G = _FSCVRT_1over2_LOWPASS_G;
  } else if (resampleRate == 3) {
    *nsect = _FSCVRT_1over3_LOWPASS_NSEC;
    for (idx = 0; idx < (*nsect); idx++) {
      B[idx] = &(_FSCVRT_1over3_LOWPASS_B[idx][0]);
      A[idx] = &(_FSCVRT_1over3_LOWPASS_A[idx][0]);
    }
    *G = _FSCVRT_1over3_LOWPASS_G;
  } else if (resampleRate == 4) {
    *nsect = _FSCVRT_1over4_LOWPASS_NSEC;
    for (idx = 0; idx < (*nsect); idx++) {
      B[idx] = &(_FSCVRT_1over4_LOWPASS_B[idx][0]);
      A[idx] = &(_FSCVRT_1over4_LOWPASS_A[idx][0]);
    }
    *G = _FSCVRT_1over4_LOWPASS_G;
  } else if (resampleRate == 6) {
    *nsect = _FSCVRT_1over6_LOWPASS_NSEC;
    for (idx = 0; idx < (*nsect); idx++) {
      B[idx] = &(_FSCVRT_1over6_LOWPASS_B[idx][0]);
      A[idx] = &(_FSCVRT_1over6_LOWPASS_A[idx][0]);
    }
    *G = _FSCVRT_1over6_LOWPASS_G;
  } else {  // unknown resample rate
    return -1;
  }

  return 0;
}

static int AUP_Fscvrt_dynamMemPrepare(FscvrtSt* stHdl, void* memPtrExt,
                                      size_t memSize) {
  char* memPtr = NULL;
  int biquadInBufMemSize = 0;
  int biquadOutBufMemSize = 0;
  int totalMemSize = 0;

  if (stHdl == NULL) {
    return -1;
  }

  biquadInBufMemSize = _FSCVRT_ALIGN8(sizeof(float) * stHdl->biquadInBufLen);
  totalMemSize += biquadInBufMemSize;

  biquadOutBufMemSize = _FSCVRT_ALIGN8(sizeof(float) * stHdl->biquadOutBufLen);
  totalMemSize += biquadOutBufMemSize;

  totalMemSize = _FSCVRT_MAX(totalMemSize, 80);

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

  stHdl->biquadInBuf = NULL;
  if (biquadInBufMemSize != 0) {
    stHdl->biquadInBuf = (float*)memPtr;
    memPtr += biquadInBufMemSize;
  }

  stHdl->biquadOutBuf = NULL;
  if (biquadOutBufMemSize != 0) {
    stHdl->biquadOutBuf = (float*)memPtr;
    memPtr += biquadOutBufMemSize;
  }

  if (((int)(memPtr - (char*)memPtrExt)) > totalMemSize) {
    return -1;
  }

  return (totalMemSize);
}

static int AUP_Fscvrt_checkStatCfg(FscvrtStaticCfg* pCfg) {
  if (pCfg == NULL) {
    return -1;
  }

  if (pCfg->inputFs != 16000 && pCfg->inputFs != 24000 &&
      pCfg->inputFs != 32000 && pCfg->inputFs != 48000) {
    return -1;
  }

  if (pCfg->outputFs != 16000 && pCfg->outputFs != 24000 &&
      pCfg->outputFs != 32000 && pCfg->outputFs != 48000) {
    return -1;
  }

  if (pCfg->stepSz > AUP_FSCVRT_MAX_INPUT_LEN || pCfg->stepSz < 1) {
    return -1;
  }

  if (pCfg->inputType != 0) {
    pCfg->inputType = 1;
  }

  if (pCfg->outputType != 0) {
    pCfg->outputType = 1;
  }

  return 0;
}

static int AUP_Fscvrt_publishStaticCfg(FscvrtSt* stHdl) {
  int tmpRatio;
  int ret;
  int maxResmplRate = 0;

  stHdl->mode = 0;
  stHdl->upSmplRate = 1;
  stHdl->downSmplRate = 1;
  if (stHdl->stCfg.inputFs != stHdl->stCfg.outputFs) {
    if (stHdl->stCfg.outputFs > stHdl->stCfg.inputFs) {
      tmpRatio = (stHdl->stCfg.outputFs / stHdl->stCfg.inputFs);
      if (stHdl->stCfg.outputFs == tmpRatio * stHdl->stCfg.inputFs) {
        stHdl->mode = 1;
        stHdl->upSmplRate = tmpRatio;
        stHdl->downSmplRate = 1;
      } else {
        stHdl->mode = 3;
        stHdl->upSmplRate = _FSCVRT_COMMON_FS / stHdl->stCfg.inputFs;
        stHdl->downSmplRate = _FSCVRT_COMMON_FS / stHdl->stCfg.outputFs;
      }
    } else {  // stHdl->stCfg.outputFs < stHdl->stCfg.inputFs
      tmpRatio = (stHdl->stCfg.inputFs / stHdl->stCfg.outputFs);
      if (stHdl->stCfg.inputFs == tmpRatio * stHdl->stCfg.outputFs) {
        stHdl->mode = 2;
        stHdl->upSmplRate = 1;
        stHdl->downSmplRate = tmpRatio;
      } else {
        stHdl->mode = 3;
        stHdl->upSmplRate = _FSCVRT_COMMON_FS / stHdl->stCfg.inputFs;
        stHdl->downSmplRate = _FSCVRT_COMMON_FS / stHdl->stCfg.outputFs;
      }
    }
  }

  if (stHdl->mode == 0) {
    stHdl->biquadInBufLen = 0;
    stHdl->biquadOutBufLen = 0;
  } else {
    stHdl->biquadInBufLen = stHdl->stCfg.stepSz * stHdl->upSmplRate;
    stHdl->biquadOutBufLen = 2 * (stHdl->stCfg.stepSz * stHdl->upSmplRate);
  }

  maxResmplRate = _FSCVRT_MAX(stHdl->upSmplRate, stHdl->downSmplRate);

  stHdl->nSec = 0;
  memset(stHdl->biquadB, 0, sizeof(stHdl->biquadB));
  memset(stHdl->biquadA, 0, sizeof(stHdl->biquadA));
  stHdl->biquadG = NULL;  // gain for each section

  if (stHdl->mode != 0) {
    ret = AUP_Fscvrt_FilterSet(maxResmplRate, &(stHdl->nSec), stHdl->biquadB,
                               stHdl->biquadA, &(stHdl->biquadG));
    if (ret < 0) {
      return -1;
    }
  }

  return 0;
}

static int AUP_Fscvrt_resetVariables(FscvrtSt* stHdl) {
  stHdl->biquadInBufCnt = 0;
  stHdl->biquadOutBufCnt = 0;

  if (stHdl->dynamMemPtr != NULL && stHdl->dynamMemSize > 0) {
    memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);
  }
  return 0;
}

// ==========================================================================================
// public APIs
// ==========================================================================================

int AUP_Fscvrt_create(void** stPtr) {
  FscvrtSt* tmpPtr;

  if (stPtr == NULL) {
    return -1;
  }
  *stPtr = (void*)malloc(sizeof(FscvrtSt));
  if (*stPtr == NULL) {
    return -1;
  }
  memset(*stPtr, 0, sizeof(FscvrtSt));

  tmpPtr = (FscvrtSt*)(*stPtr);

  tmpPtr->dynamMemPtr = NULL;
  tmpPtr->dynamMemSize = 0;

  tmpPtr->stCfg.inputFs = 24000;
  tmpPtr->stCfg.outputFs = 32000;
  tmpPtr->stCfg.stepSz = 240;    // 10ms processing step
  tmpPtr->stCfg.inputType = 0;   // short in
  tmpPtr->stCfg.outputType = 0;  // short out

  if (AUP_Biquad_create(&(tmpPtr->biquadSt)) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Fscvrt_destroy(void** stPtr) {
  FscvrtSt* stHdl;

  if (stPtr == NULL) {
    return 0;
  }

  stHdl = (FscvrtSt*)(*stPtr);
  if (stHdl == NULL) {
    return 0;
  }

  AUP_Biquad_destroy(&(stHdl->biquadSt));
  if (stHdl->dynamMemPtr != NULL) {
    free(stHdl->dynamMemPtr);
  }
  stHdl->dynamMemPtr = NULL;

  free(stHdl);
  (*stPtr) = NULL;

  return 0;
}

int AUP_Fscvrt_memAllocate(void* stPtr, const FscvrtStaticCfg* pCfg) {
  FscvrtSt* stHdl = NULL;
  FscvrtStaticCfg tmpStatCfg = {0};
  Biquad_StaticCfg bqStatCfg;
  int idx, ret;
  int totalMemSize = 0;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (FscvrtSt*)(stPtr);

  memcpy(&tmpStatCfg, pCfg, sizeof(FscvrtStaticCfg));
  if (AUP_Fscvrt_checkStatCfg(&tmpStatCfg) < 0) {
    return -1;
  }
  memcpy(&(stHdl->stCfg), &tmpStatCfg, sizeof(FscvrtStaticCfg));

  if (AUP_Fscvrt_publishStaticCfg(stHdl) < 0) {
    return -1;
  }

  // check memory requirement
  totalMemSize = AUP_Fscvrt_dynamMemPrepare(stHdl, NULL, 0);
  if (totalMemSize < 0) {
    return -1;
  }

  // allocate dynamic memory
  if ((size_t)totalMemSize > stHdl->dynamMemSize) {
    if (stHdl->dynamMemPtr != NULL) {
      free(stHdl->dynamMemPtr);
      stHdl->dynamMemSize = 0;
    }
    stHdl->dynamMemPtr = (void*)malloc(totalMemSize);
    if (stHdl->dynamMemPtr == NULL) {
      return -1;
    }
    stHdl->dynamMemSize = totalMemSize;
  }
  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);

  // setup the pointers/variable
  if (AUP_Fscvrt_dynamMemPrepare(stHdl, stHdl->dynamMemPtr,
                                 stHdl->dynamMemSize) < 0) {
    return -1;
  }

  // memAllocation for upSmplBiquadSt and downSmplBiquadSt
  if (stHdl->nSec != 0) {
    if (stHdl->nSec > AGORA_UAP_BIQUAD_MAX_SECTION) {
      return -1;
    }
    memset(&bqStatCfg, 0, sizeof(Biquad_StaticCfg));
    bqStatCfg.maxNSample = (size_t)(stHdl->biquadInBufLen);
    bqStatCfg.nsect = stHdl->nSec;
    for (idx = 0; idx < stHdl->nSec; idx++) {
      bqStatCfg.B[idx] = stHdl->biquadB[idx];
      bqStatCfg.A[idx] = stHdl->biquadA[idx];
    }
    bqStatCfg.G = stHdl->biquadG;

    ret = AUP_Biquad_memAllocate(stHdl->biquadSt, &bqStatCfg);
    if (ret < 0) {
      return -1;
    }
  }

  return 0;
}

int AUP_Fscvrt_init(void* stPtr) {
  FscvrtSt* stHdl;

  if (stPtr == NULL) {
    return -1;
  }
  stHdl = (FscvrtSt*)(stPtr);

  // clear/reset run-time variables
  if (AUP_Fscvrt_resetVariables(stHdl) < 0) {
    return -1;
  }

  // init submodules ...
  if (stHdl->biquadSt != NULL && stHdl->nSec != 0) {
    if (AUP_Biquad_init(stHdl->biquadSt) < 0) {
      return -1;
    }
  }

  return 0;
}

int AUP_Fscvrt_getStaticCfg(const void* stPtr, FscvrtStaticCfg* pCfg) {
  const FscvrtSt* stHdl;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }
  stHdl = (const FscvrtSt*)(stPtr);

  memcpy(pCfg, &(stHdl->stCfg), sizeof(FscvrtStaticCfg));

  return 0;
}

int AUP_Fscvrt_getInfor(const void* stPtr, FscvrtGetData* buff) {
  const FscvrtSt* stHdl;
  int delayBiquad = 0;
  int tmp;

  if (stPtr == NULL || buff == NULL) {
    return -1;
  }
  stHdl = (const FscvrtSt*)(stPtr);

  if (stHdl->nSec != 0) {
    if (AUP_Biquad_getAlgDelay(stHdl->biquadSt, &delayBiquad) < 0) {
      return -1;
    }
  }

  if (stHdl->mode == 0) {
    buff->delayInInputFs = 0;
  } else if (stHdl->mode == 1) {
    buff->delayInInputFs =
        (int)roundf(delayBiquad / (float)(stHdl->upSmplRate));
  } else if (stHdl->mode == 2) {  // direct downsampling
    buff->delayInInputFs = delayBiquad;
  } else {  // stHdl->mode == 3
    buff->delayInInputFs =
        (int)roundf(delayBiquad / (float)(stHdl->upSmplRate));
  }
  tmp = stHdl->stCfg.stepSz * stHdl->upSmplRate / stHdl->downSmplRate;
  if (tmp * stHdl->downSmplRate == stHdl->stCfg.stepSz * stHdl->upSmplRate) {
    buff->maxOutputStepSz = tmp;
  } else {
    buff->maxOutputStepSz = tmp + 1;
  }

  return 0;
}

int AUP_Fscvrt_proc(void* stPtr, const FscvrtInData* pIn, FscvrtOutData* pOut) {
  FscvrtSt* stHdl = NULL;
  const FscvrtStaticCfg* pCfg;
  Biquad_InputData bqdInData;
  Biquad_OutputData bqdOutData;
  const short* shortSrcPtr = NULL;
  const float* floatSrcPtr = NULL;
  short* shortTgtPtr = NULL;
  float* floatTgtPtr = NULL;
  int idx, tgtIdx;
  int nOutSamples = 0, samplesTaken = 0, samplesLeft = 0;
  int jumpRate;

  if (stPtr == NULL || pIn == NULL || pOut == NULL || pIn->inDataSeq == NULL ||
      pOut->outDataSeq == NULL) {  //  pCtrl == NULL
    return -1;
  }

  stHdl = (FscvrtSt*)(stPtr);
  pCfg = (const FscvrtStaticCfg*)&(stHdl->stCfg);
  shortSrcPtr = (const short*)(pIn->inDataSeq);
  floatSrcPtr = (const float*)(pIn->inDataSeq);
  // ==============================================================================
  // mode-0: bypass
  if (stHdl->mode == 0) {  // direct bypass
    if (pIn->outDataSeqLen < pCfg->stepSz) {
      return -1;
    }
    pOut->nOutData = pCfg->stepSz;
    pOut->outDataType = pCfg->outputType;
    if (pIn->inDataSeq == pOut->outDataSeq) {
      if (pCfg->outputType == pCfg->inputType)
        return 0;  // we don't need to do anything
      return -1;
      // if input buffer and the output buffer are the same, but required
      // different data type: error, we currently do not support such usecase
    }

    if (pCfg->inputType == 0 && pCfg->outputType == 0) {
      memcpy(pOut->outDataSeq, pIn->inDataSeq, sizeof(short) * pCfg->stepSz);
    } else if (pCfg->inputType == 1 && pCfg->outputType == 1) {
      memcpy(pOut->outDataSeq, pIn->inDataSeq, sizeof(float) * pCfg->stepSz);
    } else if (pCfg->inputType == 0 && pCfg->outputType == 1) {
      for (idx = 0; idx < pCfg->stepSz; idx++) {
        ((float*)pOut->outDataSeq)[idx] = ((short*)pIn->inDataSeq)[idx];
      }
    } else {  // if (pCfg->inputType == 1 && pCfg->outputType == 0)
      for (idx = 0; idx < pCfg->stepSz; idx++) {
        ((short*)pOut->outDataSeq)[idx] =
            (short)_FSCVRT_FLOAT2SHORT(((float*)pIn->inDataSeq)[idx]);
      }
    }

    return 0;
  }

  // prepare input buffer for Biquad .....
  memset(stHdl->biquadInBuf, 0, sizeof(float) * stHdl->biquadInBufLen);
  if (pCfg->inputType == 0) {
    for (idx = 0; idx < pCfg->stepSz; idx++) {
      stHdl->biquadInBuf[idx * (stHdl->upSmplRate)] =
          ((float)shortSrcPtr[idx]) * stHdl->upSmplRate;
    }
  } else {
    for (idx = 0; idx < pCfg->stepSz; idx++) {
      stHdl->biquadInBuf[idx * (stHdl->upSmplRate)] =
          floatSrcPtr[idx] * stHdl->upSmplRate;
    }
  }

  // biquad filtering ......
  memset(&bqdInData, 0, sizeof(Biquad_InputData));
  memset(&bqdOutData, 0, sizeof(Biquad_OutputData));
  bqdInData.samplesPtr = (const void*)(stHdl->biquadInBuf);
  bqdInData.sampleType = 1;
  bqdInData.nsamples = (size_t)(pCfg->stepSz * stHdl->upSmplRate);
  bqdOutData.outputBuff = (void*)&(stHdl->biquadOutBuf[stHdl->biquadOutBufCnt]);
  if (stHdl->biquadOutBufCnt + (pCfg->stepSz * stHdl->upSmplRate) >
      stHdl->biquadOutBufLen) {
    return -1;
  }
  if (AUP_Biquad_proc(stHdl->biquadSt, &bqdInData, &bqdOutData) < 0) {
    return -1;
  }
  stHdl->biquadOutBufCnt += (pCfg->stepSz * stHdl->upSmplRate);

  // checking the output buffer .........
  nOutSamples = stHdl->biquadOutBufCnt / stHdl->downSmplRate;
  if (pIn->outDataSeqLen < nOutSamples) {
    return -1;
  }

  // prepare output data, downsampling and throwing out ......
  pOut->nOutData = nOutSamples;
  pOut->outDataType = pCfg->outputType;

  shortTgtPtr = (short*)pOut->outDataSeq;
  floatTgtPtr = (float*)pOut->outDataSeq;
  jumpRate = stHdl->downSmplRate;
  if (pCfg->outputType == 0) {  // -> shortTgtPtr
    for (idx = (jumpRate - 1), tgtIdx = 0; idx < stHdl->biquadOutBufCnt;
         idx += jumpRate, tgtIdx++) {
      shortTgtPtr[tgtIdx] = _FSCVRT_FLOAT2SHORT(stHdl->biquadOutBuf[idx]);
    }
  } else {  // -> floatTgtPtr
    for (idx = (jumpRate - 1), tgtIdx = 0; idx < stHdl->biquadOutBufCnt;
         idx += jumpRate, tgtIdx++) {
      floatTgtPtr[tgtIdx] = stHdl->biquadOutBuf[idx];
    }
  }
  if (nOutSamples != tgtIdx) {
    return -1;
  }

  // update the stHdl->biquadOutBuf and stHdl->biquadOutBufCnt
  samplesTaken = nOutSamples * jumpRate;
  samplesLeft = stHdl->biquadOutBufCnt - samplesTaken;
  if (samplesLeft == 0) {
    stHdl->biquadOutBufCnt = 0;
  } else if (samplesLeft > 0) {
    stHdl->biquadOutBufCnt = samplesLeft;
    memmove(stHdl->biquadOutBuf, &(stHdl->biquadOutBuf[samplesTaken]),
            sizeof(float) * samplesLeft);
  } else {  // samplesLeft < 0
    stHdl->biquadOutBufCnt = 0;
    return -1;
  }

  return 0;
}
