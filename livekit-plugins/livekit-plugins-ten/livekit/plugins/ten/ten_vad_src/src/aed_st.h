//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __AED_ST_H__
#define __AED_ST_H__

#include <stdio.h>
#include <onnxruntime_c_api.h>

#include "aed.h"

#define AUP_AED_FS (16000)
#define AUP_AED_MAX_IN_BUFF_SIZE (256)
#define AUP_AED_POWER_SPCTR_NORMALIZER (9.3132e-10f)  // = 1/(32768^2)
#define AUP_AED_OUTPUT_SMOOTH_FILTER_LEN (10)         // 160ms

#define AUP_AED_MEL_FILTER_BANK_NUM (40)
#define AUP_AED_LOOKAHEAD_NFRM (1)
#define AUP_AED_CONTEXT_WINDOW_LEN (3)  // context window length of AIVAD
#define AUP_AED_FEA_LEN \
  (AUP_AED_MEL_FILTER_BANK_NUM + 1)  // feature length of AIVAD

#define AUP_AED_PITCH_EST_USE_LPC (1)
#define AUP_AED_PITCH_EST_PROCFS (4000)
#if AUP_AED_PITCH_EST_PROCFS == 2000
#define AUP_AED_PITCH_EST_DEFAULT_VOICEDTHR (0.45f)
#else
#define AUP_AED_PITCH_EST_DEFAULT_VOICEDTHR (0.4f)
#endif

#define AUP_AED_MODEL_IO_NUM (5)
#define AUP_AED_MODEL_NAME_LENGTH (32)
#define AUP_AED_MODEL_HIDDEN_DIM (64)

class AUP_MODULE_AIVAD {
 public:
  AUP_MODULE_AIVAD(char* onnx_path);
  ~AUP_MODULE_AIVAD();
  int Process(float* input, float* output);
  int Reset();

 private:
  const OrtApi* ort_api = NULL;
  OrtAllocator* ort_allocator = NULL;
  OrtEnv* ort_env = NULL;
  OrtSession* ort_session = NULL;
  int inited = 0;
  int clear_hidden = 0;

  char input_names_buf[AUP_AED_MODEL_IO_NUM][AUP_AED_MODEL_NAME_LENGTH] = {0};
  const char* input_names[AUP_AED_MODEL_IO_NUM] = {NULL};
  float input_data_buf_0[AUP_AED_CONTEXT_WINDOW_LEN * AUP_AED_FEA_LEN] = {0};
  float input_data_buf_1234[AUP_AED_MODEL_IO_NUM - 1]
                           [AUP_AED_MODEL_HIDDEN_DIM] = {0};
  OrtValue* ort_input_tensors[AUP_AED_MODEL_IO_NUM] = {NULL};

  char output_names_buf[AUP_AED_MODEL_IO_NUM][AUP_AED_MODEL_NAME_LENGTH] = {0};
  const char* output_names[AUP_AED_MODEL_IO_NUM] = {NULL};
  OrtValue* ort_output_tensors[AUP_AED_MODEL_IO_NUM] = {NULL};
};

typedef struct Aed_St_ {
  void* dynamMemPtr;    // memory pointer holding the dynamic memory
  size_t dynamMemSize;  // size of the buffer *dynamMemPtr

  Aed_StaticCfg stCfg;

  Aed_DynamCfg dynamCfg;

  // Internal Static Config Registers, which are generated from stCfg
  size_t extFftSz;  // externally decided FFT-Sz
  size_t extHopSz;  // externally decided FFT-Hop-Sz
  size_t extNBins;  // (FFTSz/2) + 1
  size_t extWinSz;  // externally decided FFT-Window-Sz

  size_t intFftSz;                 // internal FFT Sz
  size_t intHopSz;                 // internal Hop Sz
  size_t intWinSz;                 // internal Window Sz
  size_t intNBins;                 // internal NBins
  const float* intAnalyWindowPtr;  // internal analysis pointer
  int intAnalyFlag;                // whether to do internal analysis
  // 0: directly use external spectrum
  // 1: use external spectrum with interpolation / exterpolation
  // 2: need to redo analysis based on input time-domain signal
  size_t inputTimeFIFOLen;  // length of input FIFO buffer
  // if = 0: no need for input time-domain FIFO Queue

  // Internal static config registers for pitch-est module
  size_t feaSz;
  size_t melFbSz;
  size_t algDelay;  // in terms of processing frames
  size_t algCtxtSz;
  size_t frmRmsBufLen;  // frameRmsBuff: buffer-length of frameRmsBuff (FIFO)

  // Internal dynamic Config Registers, which are generated from dynamCfg
  size_t aivadResetFrmNum;
  float voiceDecideThresh;

  // SubModules
  AUP_MODULE_AIVAD* aivadInf;

  void* pitchEstStPtr;  // pitch-estimation module handler
  void* timeInAnalysis;
  // state handler of STFT analysis module

  // Variables
  int aedProcFrmCnt;  // counter of consecutive AI-VAD processed frames
  int inputTimeFIFOIdx;
  float* inputTimeFIFO;  // [inputTimeFIFOLen]
  // input fifo buffer of time-signal to adjust between extHopSz and intHopSz
  float* inputEmphTimeFIFO;     // [inputTimeFIFOLen]
  float* aivadInputCmplxSptrm;  // [intFftSz]
  float* aivadInputBinPow;      // [intNBins]  // AIVAD input power spectrum
  size_t aivadResetCnt;
  float timeSignalPre;
  float aivadScore;
  float aivadScorePre;

  float pitchFreq;      // input audio pitch in Hz
  float* frameRmsBuff;  // [frmRmsBufLen], FIFO, to delay frmRms result so that
                        // it aligns with AIVAD result
  float* aivadInputFeatStack;  // [...] = [AUP_AED_CONTEXT_WINDOW_LEN *
                               // AUP_AED_FEA_LEN]
  float* melFilterBankCoef;    // [melFbSz][nBins]
  size_t* melFilterBinBuff;    // [melFbSz + 2]
  float* inputFloatBuff;       // [hopSz]
} Aed_St;

#endif
