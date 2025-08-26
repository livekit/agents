//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include "aed.h"
#include "aed_st.h"
#include "coeff.h"
#include "pitch_est.h"
#include "stft.h"
#include <assert.h>

#define AUP_AED_ALIGN8(o) (((o) + 7) & (~7))
#define AUP_AED_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define AUP_AED_MIN(x, y) (((x) > (y)) ? (y) : (x))
#define AUP_AED_EPS (1e-20f)

/// ///////////////////////////////////////////////////////////////////////
/// Internal Utils
/// ///////////////////////////////////////////////////////////////////////

AUP_MODULE_AIVAD::AUP_MODULE_AIVAD(char* onnx_path) {
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtStatus* status =
      ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "TEN-VAD", &ort_env);
  if (status) {
    printf("Failed to create env: %s\n", ort_api->GetErrorMessage(status));
    ort_api->ReleaseStatus(status);
    ort_api->ReleaseEnv(ort_env);
    ort_env = NULL;
    return;
  }

  OrtSessionOptions* session_options;
  ort_api->CreateSessionOptions(&session_options);
  ort_api->SetIntraOpNumThreads(session_options, 1);
  status =
      ort_api->CreateSession(ort_env, onnx_path, session_options, &ort_session);
  ort_api->ReleaseSessionOptions(session_options);
  if (status) {
    printf("Failed to create ort_session: %s\n",
           ort_api->GetErrorMessage(status));
    ort_api->ReleaseStatus(status);
    ort_api->ReleaseEnv(ort_env);
    ort_env = NULL;
    return;
  }

  ort_api->GetAllocatorWithDefaultOptions(&ort_allocator);
  size_t num_inputs;
  ort_api->SessionGetInputCount(ort_session, &num_inputs);
  assert(num_inputs == AUP_AED_MODEL_IO_NUM);
  for (size_t i = 0; i < num_inputs; i++) {
    char* input_name;
    ort_api->SessionGetInputName(ort_session, i, ort_allocator, &input_name);
    strncpy(input_names_buf[i], input_name, sizeof(input_names_buf[i]));
    input_names[i] = input_names_buf[i];
    ort_api->AllocatorFree(ort_allocator, input_name);
  }

  size_t num_outputs;
  ort_api->SessionGetOutputCount(ort_session, &num_outputs);
  assert(num_outputs == AUP_AED_MODEL_IO_NUM);
  for (size_t i = 0; i < num_outputs; i++) {
    char* output_name;
    ort_api->SessionGetOutputName(ort_session, i, ort_allocator, &output_name);
    strncpy(output_names_buf[i], output_name, sizeof(output_names_buf[i]));
    output_names[i] = output_names_buf[i];
    ort_api->AllocatorFree(ort_allocator, output_name);
  }

  OrtMemoryInfo* memory_info;
  status = ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault,
                                        &memory_info);
  if (status != NULL) {
    printf("Failed to create memory info: %s\n",
           ort_api->GetErrorMessage(status));
    ort_api->ReleaseStatus(status);
    ort_api->ReleaseSession(ort_session);
    ort_api->ReleaseEnv(ort_env);
    ort_session = NULL;
    ort_env = NULL;
    return;
  }
  int64_t input_shapes0[] = {1, AUP_AED_CONTEXT_WINDOW_LEN, AUP_AED_FEA_LEN};
  int64_t input_shapes1234[] = {1, AUP_AED_MODEL_HIDDEN_DIM};
  for (int i = 0; i < num_inputs; i++) {
    status = ort_api->CreateTensorWithDataAsOrtValue(
        memory_info, i == 0 ? input_data_buf_0 : input_data_buf_1234[i - 1],
        i == 0 ? sizeof(input_data_buf_0) : sizeof(input_data_buf_1234[i - 1]),
        i == 0 ? input_shapes0 : input_shapes1234,
        i == 0 ? sizeof(input_shapes0) / sizeof(input_shapes0[0])
               : sizeof(input_shapes1234) / sizeof(input_shapes1234[0]),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_input_tensors[i]);
    if (status != NULL) {
      printf("Failed to create input tensor %d: %s\n", i,
             ort_api->GetErrorMessage(status));
      ort_api->ReleaseStatus(status);
      ort_api->ReleaseSession(ort_session);
      ort_api->ReleaseEnv(ort_env);
      ort_session = NULL;
      ort_env = NULL;
      return;
    }
  }

  int64_t output_shapes0[] = {1, 1, 1};
  int64_t output_shapes1234[] = {1, AUP_AED_MODEL_HIDDEN_DIM};
  for (int i = 0; i < num_outputs; i++) {
    status = ort_api->CreateTensorAsOrtValue(
        ort_allocator, i == 0 ? output_shapes0 : output_shapes1234,
        i == 0 ? sizeof(output_shapes0) / sizeof(output_shapes0[0])
               : sizeof(output_shapes1234) / sizeof(output_shapes1234[0]),
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_output_tensors[i]);
    if (status != NULL) {
      printf("Failed to create output tensor %d: %s\n", i,
             ort_api->GetErrorMessage(status));
      ort_api->ReleaseStatus(status);
      ort_api->ReleaseSession(ort_session);
      ort_api->ReleaseEnv(ort_env);
      ort_session = NULL;
      ort_env = NULL;
      return;
    }
  }
  inited = 1;
}

AUP_MODULE_AIVAD::~AUP_MODULE_AIVAD() {
  for (int i = 0; i < AUP_AED_MODEL_IO_NUM; i++) {
    if (ort_output_tensors[i]) {
      ort_api->ReleaseValue(ort_output_tensors[i]);
    }
  }
  if (ort_session) {
    ort_api->ReleaseSession(ort_session);
  }
  if (ort_env) {
    ort_api->ReleaseEnv(ort_env);
  }
}

int AUP_MODULE_AIVAD::Process(float* input, float* output) {
  if (!inited) {
    printf("not inited!\n");
    return -1;
  }

  memcpy(input_data_buf_0, input, sizeof(input_data_buf_0));
  if (clear_hidden) {
    memset(input_data_buf_1234, 0, sizeof(input_data_buf_1234));
    clear_hidden = 0;
  }
  OrtStatus* status = ort_api->Run(
      ort_session, NULL, input_names, ort_input_tensors, AUP_AED_MODEL_IO_NUM,
      output_names, AUP_AED_MODEL_IO_NUM, ort_output_tensors);
  float* output_data;
  ort_api->GetTensorMutableData(ort_output_tensors[0], (void**)&output_data);
  *output = output_data[0];
  for (int i = 1; i < AUP_AED_MODEL_IO_NUM; i++) {
    ort_api->GetTensorMutableData(ort_output_tensors[i], (void**)&output_data);
    memcpy(input_data_buf_1234[i - 1], output_data,
           sizeof(input_data_buf_1234[i - 1]));
  }

  return 0;
}

int AUP_MODULE_AIVAD::Reset() {
  if (!inited) {
    return -1;
  }

  clear_hidden = 1;
  return 0;
}

static int AUP_Aed_checkStatCfg(Aed_StaticCfg* pCfg) {
  if (pCfg == NULL) {
    return -1;
  }

#if AUP_AED_FEA_LEN < AUP_AED_MEL_FILTER_BANK_NUM
  return -1;
#endif

  if (pCfg->hopSz < 32) {
    return -1;
  }

  if (pCfg->frqInputAvailableFlag == 1) {
    if (pCfg->fftSz < 128 || pCfg->fftSz < pCfg->hopSz) {
      return -1;
    }
    if (pCfg->anaWindowSz > pCfg->fftSz || pCfg->anaWindowSz < pCfg->hopSz) {
      return -1;
    }
  }

  return 0;
}

static int AUP_Aed_publishStaticCfg(Aed_St* stHdl) {
  const Aed_StaticCfg* pStatCfg;

  if (stHdl == NULL) {
    return -1;
  }
  pStatCfg = (const Aed_StaticCfg*)(&(stHdl->stCfg));

  stHdl->extFftSz = 0;
  stHdl->extNBins = 0;
  stHdl->extWinSz = 0;
  if (pStatCfg->frqInputAvailableFlag == 1) {
    stHdl->extFftSz = pStatCfg->fftSz;
    stHdl->extNBins = (stHdl->extFftSz >> 1) + 1;
    stHdl->extWinSz = pStatCfg->anaWindowSz;
  }
  stHdl->extHopSz = pStatCfg->hopSz;

  stHdl->intFftSz = AUP_AED_ASSUMED_FFTSZ;
  stHdl->intHopSz = AUP_AED_ASSUMED_HOPSZ;
  stHdl->intWinSz = AUP_AED_ASSUMED_WINDOWSZ;
  stHdl->intNBins = (stHdl->intFftSz >> 1) + 1;
  stHdl->intAnalyWindowPtr = AUP_AED_STFTWindow_Hann768;

  if (pStatCfg->frqInputAvailableFlag == 0 ||
      stHdl->extHopSz != stHdl->intHopSz) {
    // external STFT analysis framework is not supported at all
    stHdl->intAnalyFlag =
        2;  // internally redo analysis based on input time signal
  } else if (stHdl->extFftSz == stHdl->intFftSz) {
    // external STFT analysis framework completely match with internal
    // requirement
    stHdl->intAnalyFlag = 0;  // directly use external spectrum
  } else {  // external spectrum need to be interpolated or extrapolated before
            // AIVAD
    stHdl->intAnalyFlag =
        1;  // use external spectrum with interpolation / exterpolation
  }
  stHdl->inputTimeFIFOLen = stHdl->extHopSz + stHdl->intHopSz;

  // for aiaed release2.0.0, pre-emphasis for input time-signal is needed,
  // therefore, we need redo analysis based on input time signal preprocessed by
  // pre-emphasis.
  stHdl->intAnalyFlag =
      2;  // internally redo analysis based on input time signal

  stHdl->feaSz = (size_t)AUP_AED_FEA_LEN;
  stHdl->melFbSz = (size_t)AUP_AED_MEL_FILTER_BANK_NUM;
  stHdl->algDelay = (size_t)AUP_AED_LOOKAHEAD_NFRM;
  stHdl->algCtxtSz = (size_t)AUP_AED_CONTEXT_WINDOW_LEN;
  stHdl->frmRmsBufLen = AUP_AED_MAX(1, stHdl->algDelay);

  return 0;
}

static int AUP_Aed_publishDynamCfg(Aed_St* stHdl) {
  const Aed_DynamCfg* pDynmCfg;
  PE_DynamCfg peDynmCfg;
  if (stHdl == NULL) {
    return -1;
  }

  pDynmCfg = (const Aed_DynamCfg*)(&(stHdl->dynamCfg));
  stHdl->aivadResetFrmNum = pDynmCfg->resetFrameNum;
  stHdl->voiceDecideThresh = pDynmCfg->extVoiceThr;

  if (stHdl->pitchEstStPtr != NULL) {
    peDynmCfg.voicedThr = pDynmCfg->pitchEstVoicedThr;
    AUP_PE_setDynamCfg(stHdl->pitchEstStPtr, &peDynmCfg);
  }

  return 0;
}

static int AUP_Aed_resetVariables(Aed_St* stHdl) {
  if (stHdl == NULL) {
    return -1;
  }

  // first clear all the dynamic memory, all the dynamic variables which are
  // not listed bellow are cleared to 0 by this step
  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);

  float* melFbCoef = stHdl->melFilterBankCoef;
  size_t* melBinBuff = stHdl->melFilterBinBuff;
  size_t i, j;
  size_t nBins = stHdl->intNBins;
  size_t melFbSz = stHdl->melFbSz;

  stHdl->aedProcFrmCnt = 0;
  stHdl->inputTimeFIFOIdx = 0;
  stHdl->aivadResetCnt = 0;
  stHdl->timeSignalPre = 0.0f;
  stHdl->aivadScore =
      -1.0f;  // as default value, labeling as aed is not working yet
  stHdl->aivadScorePre = -1.0f;

  stHdl->pitchFreq = 0.0f;

  // generate mel filter-bank coefficients
  float low_mel = 2595.0f * log10f(1.0f + 0.0f / 700.0f);
  float high_mel = 2595.0f * log10f(1.0f + 8000.0f / 700.0f);
  float mel_points = 0.0f;
  float hz_points = 0.0f;
  size_t idx = 0;

  for (i = 0; i < melFbSz + 2; i++) {
    mel_points = i * (high_mel - low_mel) / ((float)melFbSz + 1.0f) + low_mel;
    hz_points = 700.0f * (powf(10.0f, mel_points / 2595.0f) - 1.0f);
    melBinBuff[i] =
        (size_t)((stHdl->intFftSz + 1.0f) * hz_points / (float)AUP_AED_FS);
    if (i > 0 && melBinBuff[i] == melBinBuff[i - 1]) {
      return -1;
    }
  }

  for (j = 0; j < melFbSz; j++) {
    for (i = melBinBuff[j]; i < melBinBuff[j + 1]; i++) {
      idx = j * nBins + i;
      melFbCoef[idx] = (float)(i - melBinBuff[j]) /
                       (float)(melBinBuff[j + 1] - melBinBuff[j]);
    }
    for (i = melBinBuff[j + 1]; i < melBinBuff[j + 2]; i++) {
      idx = j * nBins + i;
      melFbCoef[idx] = (float)(melBinBuff[j + 2] - i) /
                       (float)(melBinBuff[j + 2] - melBinBuff[j + 1]);
    }
  }

  if (stHdl->pitchEstStPtr != NULL) {
    if (AUP_PE_init(stHdl->pitchEstStPtr) < 0) {
      return -1;
    }
  }

  if (stHdl->aivadInf != NULL) {
    stHdl->aivadInf->Reset();
  }

  if (stHdl->timeInAnalysis != NULL) {
    if (AUP_Analyzer_init(stHdl->timeInAnalysis) < 0) {
      return -1;
    }
  }

  return 0;
}

static int AUP_Aed_addOneCnter(int cnter) {
  cnter++;
  if (cnter >= 1000000000) {
    cnter = 0;  // reset every half year
  }
  return (cnter);
}

static void AUP_Aed_binPowerConvert(const float* src, float* tgt, int srcNBins,
                                    int tgtNBins) {
  float rate;
  int srcIdx, tgtIdx;
  if (srcNBins == tgtNBins) {
    memcpy(tgt, src, sizeof(float) * tgtNBins);
    return;
  }

  memset(tgt, 0, sizeof(float) * tgtNBins);

  rate = (float)(srcNBins - 1) / (float)(tgtNBins - 1);
  for (tgtIdx = 0; tgtIdx < tgtNBins; tgtIdx++) {
    srcIdx = (int)(tgtIdx * rate);
    srcIdx = AUP_AED_MIN(srcNBins - 1, AUP_AED_MAX(srcIdx, 0));
    tgt[tgtIdx] = src[srcIdx];
  }

  return;
}

static void AUP_Aed_CalcBinPow(int nBins, const float* cmplxSpctr,
                               float* binPow) {
  int idx, realIdx, imagIdx;

  // bin-0
  binPow[0] = cmplxSpctr[0] * cmplxSpctr[0];

  // bin-(NBins-1)
  binPow[nBins - 1] = cmplxSpctr[1] * cmplxSpctr[1];

  for (idx = 1; idx < (nBins - 1); idx++) {
    realIdx = idx << 1;
    imagIdx = realIdx + 1;

    binPow[idx] = cmplxSpctr[realIdx] * cmplxSpctr[realIdx] +
                  cmplxSpctr[imagIdx] * cmplxSpctr[imagIdx];
  }
  return;
}

static int AUP_Aed_pitch_proc(void* pitchModule, const float* timeSignal,
                              size_t timeLen, const float* binPow, size_t nBins,
                              PE_OutputData* pOut) {
  PE_InputData peInData;

  peInData.timeSignal = timeSignal;
  peInData.hopSz = (int)timeLen;
  peInData.inBinPow = binPow;
  peInData.nBins = (int)nBins;
  pOut->pitchFreq = 0;
  pOut->voiced = -1;
  return AUP_PE_proc(pitchModule, &peInData, pOut);
}

static int AUP_Aed_aivad_proc(Aed_St* stHdl, const float* inBinPow,
                              float* aivadScore) {
  if (stHdl == NULL || inBinPow == NULL || aivadScore == NULL) {
    return -1;
  }

  size_t i, j;
  size_t nBins = stHdl->intNBins;
  size_t melFbSz = stHdl->melFbSz;
  size_t srcOffset;
  size_t srcLen;

  float* aivadInputFeatStack = stHdl->aivadInputFeatStack;
  float* melFbCoef = stHdl->melFilterBankCoef;
  const float* aivadFeatMean = AUP_AED_FEATURE_MEANS;
  const float* aivadFeatStd = AUP_AED_FEATURE_STDS;
  float* curMelFbCoefPtr = NULL;
  float* curInputFeatPtr = NULL;
  float perBandValue = 0.0f;
  float powerNormal = 32768.0f * 32768.0f;

  // update aivad feature buff.
  srcOffset = stHdl->feaSz;
  srcLen = (stHdl->algCtxtSz - 1) * stHdl->feaSz;
  memmove(aivadInputFeatStack, aivadInputFeatStack + srcOffset,
          sizeof(float) * srcLen);
  curInputFeatPtr = aivadInputFeatStack + srcLen;

  // cal. mel-filter-bank feature
  for (i = 0; i < melFbSz; i++) {
    perBandValue = 0.0f;
    curMelFbCoefPtr = melFbCoef + i * nBins;
    for (j = 0; j < nBins; j++) {
      perBandValue += (inBinPow[j] * curMelFbCoefPtr[j]);
    }
    perBandValue = perBandValue / powerNormal;
    perBandValue = logf(perBandValue + AUP_AED_EPS);
    curInputFeatPtr[i] =
        (perBandValue - aivadFeatMean[i]) / (aivadFeatStd[i] + AUP_AED_EPS);
  }

  // extra feat.
  for (i = melFbSz; i < stHdl->feaSz; i++) {
    curInputFeatPtr[i] =
        (stHdl->pitchFreq - aivadFeatMean[i]) / (aivadFeatStd[i] + AUP_AED_EPS);
  }

  // exe. aivad
  // exe. aivad
  float aivadOutput;
  if (stHdl->aivadInf != NULL &&
      stHdl->aivadInf->Process(stHdl->aivadInputFeatStack, &aivadOutput) != 0) {
    return -1;
  }

  (*aivadScore) = aivadOutput;

  stHdl->aivadResetCnt += 1;
  if (stHdl->aivadResetCnt >= stHdl->aivadResetFrmNum) {
    if (stHdl->aivadInf != NULL && stHdl->aivadInf->Reset() != 0) {
    }
    stHdl->aivadResetCnt = 0;
  }

  return 0;
}

static int AUP_Aed_dynamMemPrepare(Aed_St* stHdl, void* memPtrExt,
                                   size_t memSize) {
  if (stHdl == NULL) {
    return -1;
  }
  size_t pitchInNBins = stHdl->intNBins;
  size_t totalMemSize = 0;
  size_t inputTimeFIFOMemSize = 0;
  size_t inputEmphTimeFIFOMemSize = 0;
  size_t aivadInputCmplxSptrmMemSize = 0;
  size_t aivadInputBinPowMemSize = 0;
  size_t frameRmsBuffMemSize = 0;
  size_t aivadInputFeatStackMemSize = 0;
  size_t aimdInputFeatStackMemSize = 0;
  size_t melFilterBankCoefMemSize = 0;
  size_t melFilterBinBuffMemSize = 0;
  size_t inputFloatBuffMemSize = 0;

  // size_t vadScoreOutputBuffDelaySample = 384; // buff. delay for output
  char* memPtr = NULL;

  // size_t nBinsBufferMemSize = AUP_AED_ALIGN8(sizeof(float) * nBins);
  // size_t spctrmMemSize = AUP_AED_ALIGN8(sizeof(float) * (nBins - 1) * 2);

  inputTimeFIFOMemSize =
      AUP_AED_ALIGN8(sizeof(float) * stHdl->inputTimeFIFOLen);
  totalMemSize += inputTimeFIFOMemSize;

  inputEmphTimeFIFOMemSize =
      AUP_AED_ALIGN8(sizeof(float) * stHdl->inputTimeFIFOLen);
  totalMemSize += inputEmphTimeFIFOMemSize;

  aivadInputCmplxSptrmMemSize = AUP_AED_ALIGN8(sizeof(float) * stHdl->intFftSz);
  totalMemSize += aivadInputCmplxSptrmMemSize;

  aivadInputBinPowMemSize = AUP_AED_ALIGN8(sizeof(float) * stHdl->intNBins);
  totalMemSize += aivadInputBinPowMemSize;

  aivadInputFeatStackMemSize =
      AUP_AED_ALIGN8(sizeof(float) * stHdl->algCtxtSz * stHdl->feaSz);
  totalMemSize += aivadInputFeatStackMemSize;

  aimdInputFeatStackMemSize =
      AUP_AED_ALIGN8(sizeof(float) * stHdl->algCtxtSz * stHdl->feaSz);
  totalMemSize += aimdInputFeatStackMemSize;

  melFilterBankCoefMemSize =
      AUP_AED_ALIGN8(sizeof(float) * pitchInNBins * stHdl->feaSz);
  totalMemSize += melFilterBankCoefMemSize;

  melFilterBinBuffMemSize = AUP_AED_ALIGN8(sizeof(size_t) * (stHdl->feaSz + 2));
  totalMemSize += melFilterBinBuffMemSize;

  frameRmsBuffMemSize = AUP_AED_ALIGN8(stHdl->frmRmsBufLen * sizeof(float));
  totalMemSize += frameRmsBuffMemSize;

  inputFloatBuffMemSize = AUP_AED_ALIGN8(stHdl->extHopSz * sizeof(float));
  totalMemSize += inputFloatBuffMemSize;

  if (memPtrExt == NULL) {
    return ((int)totalMemSize);
  }

  if (totalMemSize > memSize) {
    return -1;
  }

  memPtr = (char*)memPtrExt;

  stHdl->inputTimeFIFO = (float*)memPtr;
  memPtr += inputTimeFIFOMemSize;

  stHdl->inputEmphTimeFIFO = (float*)memPtr;
  memPtr += inputEmphTimeFIFOMemSize;

  stHdl->aivadInputCmplxSptrm = (float*)memPtr;
  memPtr += aivadInputCmplxSptrmMemSize;

  stHdl->aivadInputBinPow = (float*)memPtr;
  memPtr += aivadInputBinPowMemSize;

  stHdl->aivadInputFeatStack = (float*)memPtr;
  memPtr += aivadInputFeatStackMemSize;

  stHdl->melFilterBankCoef = (float*)memPtr;
  memPtr += melFilterBankCoefMemSize;

  stHdl->melFilterBinBuff = (size_t*)memPtr;
  memPtr += melFilterBinBuffMemSize;

  stHdl->frameRmsBuff = (float*)memPtr;
  memPtr += frameRmsBuffMemSize;

  stHdl->inputFloatBuff = (float*)memPtr;
  memPtr += inputFloatBuffMemSize;

  if (((size_t)(memPtr - (char*)memPtrExt)) > totalMemSize) {
    return -1;
  }

  return ((int)totalMemSize);
}

static int AUP_Aed_runOneFrm(Aed_St* stHdl, const float* tSignal, int hopSz,
                             const float* binPowPtr, int nBins) {
  PE_OutputData peOutData = {0, 0};
  float aivadScore = -1.0f;
  float mediaFilterout = 0;
  int mediaIdx = (int)(AUP_AED_OUTPUT_SMOOTH_FILTER_LEN) / 2;
  int i;

  if (AUP_Aed_pitch_proc(stHdl->pitchEstStPtr, tSignal, hopSz, binPowPtr, nBins,
                         &peOutData) < 0) {
    return -1;
  }
  stHdl->pitchFreq = peOutData.pitchFreq;
  if (AUP_Aed_aivad_proc(stHdl, binPowPtr, &aivadScore) < 0) {
    return -1;
  }
  stHdl->aivadScore = aivadScore;

  return 0;
}

/// ///////////////////////////////////////////////////////////////////////
/// Public API
/// ///////////////////////////////////////////////////////////////////////

int AUP_Aed_create(void** stPtr) {
  if (stPtr == NULL) {
    return -1;
  }
  Aed_St* tmpPtr = (Aed_St*)malloc(sizeof(Aed_St));
  if (tmpPtr == NULL) {
    return -1;
  }
  memset(tmpPtr, 0, sizeof(Aed_St));

  if (AUP_PE_create(&(tmpPtr->pitchEstStPtr)) < 0) {
    return -1;
  }
  if (AUP_Analyzer_create(&(tmpPtr->timeInAnalysis)) < 0) {
    return -1;
  }

  tmpPtr->stCfg.enableFlag = 1;  // as default, module enabled
  tmpPtr->stCfg.fftSz = 1024;
  tmpPtr->stCfg.hopSz = 256;
  tmpPtr->stCfg.anaWindowSz = 768;
  tmpPtr->stCfg.frqInputAvailableFlag = 0;

  tmpPtr->dynamCfg.extVoiceThr = 0.5f;
  tmpPtr->dynamCfg.extMusicThr = 0.5f;
  tmpPtr->dynamCfg.extEnergyThr = 10.0f;
  tmpPtr->dynamCfg.resetFrameNum = 1875;  // TODO
  tmpPtr->dynamCfg.pitchEstVoicedThr = AUP_AED_PITCH_EST_DEFAULT_VOICEDTHR;

  (*stPtr) = (void*)tmpPtr;

  return 0;
}

int AUP_Aed_destroy(void** stPtr) {
  if (stPtr == NULL || (*stPtr) == NULL) {
    return -1;
  }
  Aed_St* stHdl = (Aed_St*)(*stPtr);

  if (stHdl->aivadInf != NULL) {
    delete stHdl->aivadInf;
  }
  stHdl->aivadInf = NULL;

  if (AUP_PE_destroy(&(stHdl->pitchEstStPtr)) < 0) {
    return -1;
  }
  if (AUP_Analyzer_destroy(&(stHdl->timeInAnalysis)) < 0) {
    return -1;
  }

  if (stHdl->dynamMemPtr != NULL) {
    free(stHdl->dynamMemPtr);
  }
  stHdl->dynamMemPtr = NULL;

  if (stHdl != NULL) {
    free(stHdl);
  }
  (*stPtr) = NULL;

  return 0;
}

int AUP_Aed_memAllocate(void* stPtr, const Aed_StaticCfg* pCfg) {
  Aed_St* stHdl = (Aed_St*)(stPtr);
  Aed_StaticCfg aedStatCfg;
  PE_StaticCfg pitchStatCfg;
  Analyzer_StaticCfg analyzerStatCfg;
  int totalMemSize = 0;

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }

  // 1th: check static cfg.
  memcpy(&aedStatCfg, pCfg, sizeof(Aed_StaticCfg));
  if (AUP_Aed_checkStatCfg(&aedStatCfg) < 0) {
    return -1;
  }

  memcpy(&(stHdl->stCfg), &aedStatCfg, sizeof(Aed_StaticCfg));

  // 2th: publish static configuration to internal statical configuration
  // registers
  if (AUP_Aed_publishStaticCfg(stHdl) < 0) {
    return -1;
  }

  // 3th: create aivad instance
  if (stHdl->aivadInf == NULL) {
    stHdl->aivadInf = new AUP_MODULE_AIVAD("onnx_model/ten-vad.onnx");
    if (stHdl->aivadInf == NULL) {
      return -1;
    }
  }
  stHdl->aivadInf->Reset();

  // 4th: memAllocate operation for Pitch-Estimator ............
  if (AUP_PE_getStaticCfg(stHdl->pitchEstStPtr, &pitchStatCfg) < 0) {
    return -1;
  }
  pitchStatCfg.fftSz = stHdl->intFftSz;
  pitchStatCfg.anaWindowSz = stHdl->intWinSz;
  pitchStatCfg.hopSz = stHdl->intHopSz;
  pitchStatCfg.useLPCPreFiltering = AUP_AED_PITCH_EST_USE_LPC;
  pitchStatCfg.procFs = AUP_AED_PITCH_EST_PROCFS;
  if (AUP_PE_memAllocate(stHdl->pitchEstStPtr, &pitchStatCfg) < 0) {
    return -1;
  }

  // creation and initialization with time-analysis module ......
  AUP_Analyzer_getStaticCfg(stHdl->timeInAnalysis, &analyzerStatCfg);
  analyzerStatCfg.win_len = (int)stHdl->intWinSz;
  analyzerStatCfg.hop_size = (int)stHdl->intHopSz;
  analyzerStatCfg.fft_size = (int)stHdl->intFftSz;
  analyzerStatCfg.ana_win_coeff = stHdl->intAnalyWindowPtr;
  if (AUP_Analyzer_memAllocate(stHdl->timeInAnalysis, &analyzerStatCfg) < 0) {
    return -1;
  }

  // 5th: check memory requirement ..............................
  totalMemSize = AUP_Aed_dynamMemPrepare(stHdl, NULL, 0);
  if (totalMemSize < 0) {
    return -1;
  }

  // 6th: allocate dynamic memory
  if (totalMemSize > (int)stHdl->dynamMemSize) {
    if (stHdl->dynamMemPtr != NULL) {
      free(stHdl->dynamMemPtr);
      stHdl->dynamMemPtr = NULL;
      stHdl->dynamMemSize = 0;
    }
    stHdl->dynamMemPtr = malloc(totalMemSize);
    if (stHdl->dynamMemPtr == NULL) {
      return -1;
    }
    stHdl->dynamMemSize = totalMemSize;
  }
  memset(stHdl->dynamMemPtr, 0, stHdl->dynamMemSize);

  // 7th: setup the pointers/variable
  if (AUP_Aed_dynamMemPrepare(stHdl, stHdl->dynamMemPtr, stHdl->dynamMemSize) <
      0) {
    return -1;
  }

  // 8th: publish internal dynamic config registers
  if (AUP_Aed_publishDynamCfg(stHdl) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Aed_init(void* stPtr) {
  Aed_St* stHdl = (Aed_St*)(stPtr);
  if (stPtr == NULL) {
    return -1;
  }

  // publish internal dynamic config registers
  if (AUP_Aed_publishDynamCfg(stHdl) < 0) {
    return -1;
  }

  // clear/reset run-time variables
  if (AUP_Aed_resetVariables(stHdl) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Aed_setDynamCfg(void* stPtr, const Aed_DynamCfg* pCfg) {
  Aed_St* stHdl = (Aed_St*)(stPtr);

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }

  memcpy(&(stHdl->dynamCfg), pCfg, sizeof(Aed_DynamCfg));

  // publish internal dynamic configuration registers
  if (AUP_Aed_publishDynamCfg(stHdl) < 0) {
    return -1;
  }

  return 0;
}

int AUP_Aed_getStaticCfg(const void* stPtr, Aed_StaticCfg* pCfg) {
  const Aed_St* stHdl = (const Aed_St*)(stPtr);

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }

  memcpy(pCfg, &(stHdl->stCfg), sizeof(Aed_StaticCfg));

  return 0;
}

int AUP_Aed_getDynamCfg(const void* stPtr, Aed_DynamCfg* pCfg) {
  const Aed_St* stHdl = (const Aed_St*)(stPtr);

  if (stPtr == NULL || pCfg == NULL) {
    return -1;
  }

  memcpy(pCfg, &(stHdl->dynamCfg), sizeof(Aed_DynamCfg));

  return 0;
}

int AUP_Aed_getAlgDelay(const void* stPtr, int* delayInFrms) {
  const Aed_St* stHdl = (const Aed_St*)(stPtr);

  if (stPtr == NULL || delayInFrms == NULL) {
    return -1;
  }

  (*delayInFrms) = (int)stHdl->algDelay;

  return 0;
}

int AUP_Aed_proc(void* stPtr, const Aed_InputData* pIn, Aed_OutputData* pOut) {
  Analyzer_InputData analyzerInput;
  Analyzer_OutputData analyzerOutput;
  Aed_St* stHdl = (Aed_St*)(stPtr);

  const float* binPowPtr = NULL;
  float frameRms = 0.0f;
  float frameEnergy = 0.0f;
  float powerNormal = 32768.0f * 32768.0f;
  int idx;

  if (stPtr == NULL) {
    return -1;
  }
  if (stHdl->stCfg.enableFlag == 0) {  // this module is disabled
    return 0;
  }
  if (pIn == NULL || pIn->timeSignal == NULL || pOut == NULL) {
    return -1;
  }

  if (stHdl->intAnalyFlag != 2) {  // the external spectra is going to be used
    if (pIn->binPower == NULL) {
      return -1;
    }
    if (pIn->nBins != (int)((stHdl->stCfg.fftSz >> 1) + 1) ||
        pIn->hopSz != (int)(stHdl->stCfg.hopSz)) {
      return -1;
    }
  }

  // cal. input frame energy ....
  for (idx = 0; idx < pIn->hopSz; idx++) {
    frameRms += (pIn->timeSignal[idx] * pIn->timeSignal[idx]);
  }
  frameEnergy = frameRms;
  frameRms = sqrtf(frameRms / (float)pIn->hopSz);
  memmove(stHdl->frameRmsBuff, stHdl->frameRmsBuff + 1,
          sizeof(float) * (stHdl->frmRmsBufLen - 1));
  stHdl->frameRmsBuff[stHdl->frmRmsBufLen - 1] = frameRms;

  // input signal conversion .........
  if ((stHdl->inputTimeFIFOIdx + pIn->hopSz) > (int)stHdl->inputTimeFIFOLen) {
    return -1;
  }

  // update pre-emphasis time signal FIFO
  float* timeSigEphaPtr = stHdl->inputEmphTimeFIFO + stHdl->inputTimeFIFOIdx;
  for (idx = 0; idx < pIn->hopSz; idx++) {
    timeSigEphaPtr[idx] = pIn->timeSignal[idx] - 0.97f * stHdl->timeSignalPre;
    stHdl->timeSignalPre = pIn->timeSignal[idx];
  }

  memcpy(stHdl->inputTimeFIFO + stHdl->inputTimeFIFOIdx, pIn->timeSignal,
         sizeof(float) * (pIn->hopSz));
  stHdl->inputTimeFIFOIdx += pIn->hopSz;

  if (stHdl->intAnalyFlag == 0) {  // directly use external spectra
    if (stHdl->inputTimeFIFOIdx != (int)(stHdl->intHopSz) ||
        (int)(stHdl->intNBins) != pIn->nBins) {
      return -1;
    }

    // one-time processing ...
    stHdl->aedProcFrmCnt = AUP_Aed_addOneCnter(stHdl->aedProcFrmCnt);
    binPowPtr = pIn->binPower;

    // update: stHdl->pitchFreq, stHdl->aivadScore
    if (AUP_Aed_runOneFrm(stHdl, stHdl->inputTimeFIFO, (int)stHdl->intHopSz,
                          binPowPtr, (int)stHdl->intNBins) < 0) {
      return -1;
    }

    // update the inputTimeFIFO
    stHdl->inputTimeFIFOIdx = 0;
  } else if (stHdl->intAnalyFlag ==
             1) {  // do interpolation or extrapolation with external spectra
    if (stHdl->inputTimeFIFOIdx != (int)(stHdl->intHopSz) ||
        (int)(stHdl->extNBins) != pIn->nBins) {
      return -1;
    }

    // one-time processing ....
    stHdl->aedProcFrmCnt = AUP_Aed_addOneCnter(stHdl->aedProcFrmCnt);
    AUP_Aed_binPowerConvert(pIn->binPower, stHdl->aivadInputBinPow,
                            (int)stHdl->extNBins, (int)stHdl->intNBins);
    binPowPtr = stHdl->aivadInputBinPow;

    // update: stHdl->pitchFreq, stHdl->aivadScore
    if (AUP_Aed_runOneFrm(stHdl, stHdl->inputTimeFIFO, (int)stHdl->intHopSz,
                          binPowPtr, (int)stHdl->intNBins) < 0) {
      return -1;
    }

    // update the inputTimeFIFO
    stHdl->inputTimeFIFOIdx = 0;
  } else {  // we need to do STFT on the input time-signal
    if (stHdl->timeInAnalysis == NULL) {
      return -1;
    }

    // loop processing .....
    while (stHdl->inputTimeFIFOIdx >= (int)stHdl->intHopSz) {
      stHdl->aedProcFrmCnt = AUP_Aed_addOneCnter(stHdl->aedProcFrmCnt);

      analyzerInput.input = stHdl->inputEmphTimeFIFO;
      analyzerInput.iLength = (int)stHdl->intHopSz;
      analyzerOutput.output = stHdl->aivadInputCmplxSptrm;
      analyzerOutput.oLength = (int)stHdl->intFftSz;
      if (AUP_Analyzer_proc(stHdl->timeInAnalysis, &analyzerInput,
                            &analyzerOutput) < 0) {
        return -1;
      }

      AUP_Aed_CalcBinPow((int)stHdl->intNBins, stHdl->aivadInputCmplxSptrm,
                         stHdl->aivadInputBinPow);
      binPowPtr = stHdl->aivadInputBinPow;

      // update: stHdl->pitchFreq, stHdl->aivadScore
      if (AUP_Aed_runOneFrm(stHdl, stHdl->inputTimeFIFO, (int)stHdl->intHopSz,
                            binPowPtr, (int)stHdl->intNBins) < 0) {
        return -1;
      }

      // update the inputTimeFIFO & inputEmphTimeFIFO.....
      if (stHdl->inputTimeFIFOIdx > (int)stHdl->intHopSz) {
        memcpy(stHdl->inputTimeFIFO, stHdl->inputTimeFIFO + stHdl->intHopSz,
               sizeof(float) * (stHdl->inputTimeFIFOIdx - stHdl->intHopSz));
        memcpy(stHdl->inputEmphTimeFIFO,
               stHdl->inputEmphTimeFIFO + stHdl->intHopSz,
               sizeof(float) * (stHdl->inputTimeFIFOIdx - stHdl->intHopSz));
      }
      stHdl->inputTimeFIFOIdx -= (int)stHdl->intHopSz;
    }
  }

  // write to output res.
  pOut->frameEnergy = frameEnergy / powerNormal;
  pOut->frameRms = stHdl->frameRmsBuff[0];
  pOut->pitchFreq = stHdl->pitchFreq;
  pOut->voiceProb = stHdl->aivadScore;
  if (pOut->voiceProb < 0.0f) {
    pOut->vadRes = -1;
  } else if (pOut->voiceProb <= stHdl->voiceDecideThresh) {
    pOut->vadRes = 0;
  } else {
    pOut->vadRes = 1;
  }

  return 0;
}
