//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#include <cassert>
#include "ten_vad.h"
#include "aed_st.h"
#include "aed.h"

static void int16_to_float(const int16_t* inputs, int inputLen, float* output) {
  for (int i = 0; i < inputLen; ++i) {
    output[i] = float(inputs[i]);
  }
}

int ten_vad_create(ten_vad_handle_t* handle, size_t hop_size, float threshold) {
  if (AUP_Aed_create(handle) < 0) {
    return -1;
  }
  Aed_St* stHdl = nullptr;
  Aed_StaticCfg aedStCfg;
  aedStCfg.enableFlag = 1;
  aedStCfg.fftSz = 0;
  aedStCfg.hopSz = hop_size;
  aedStCfg.anaWindowSz = 0;
  aedStCfg.frqInputAvailableFlag = 0;
  stHdl = (Aed_St*)(*handle);
  stHdl->dynamCfg.extVoiceThr = threshold;

  if (AUP_Aed_memAllocate(*handle, &aedStCfg) < 0) {
    return -1;
  }
  if (AUP_Aed_init(*handle) < 0) {
    return -1;
  }
  return 0;
}

int ten_vad_process(ten_vad_handle_t handle, const int16_t* audio_data,
                    size_t audio_data_length, float* out_probability,
                    int* out_flag) {
  if (handle == nullptr || audio_data == nullptr ||
      out_probability == nullptr || out_flag == nullptr) {
    return -1;
  }
  Aed_St* ptr = (Aed_St*)handle;
  assert(audio_data_length == ptr->stCfg.hopSz);
  int16_to_float(audio_data, audio_data_length, ptr->inputFloatBuff);
  Aed_InputData aedInputData;
  Aed_OutputData aedOutputData;
  aedInputData.binPower = NULL;
  aedInputData.hopSz = ptr->stCfg.hopSz;
  aedInputData.nBins = -1;
  aedInputData.timeSignal = ptr->inputFloatBuff;
  int ret = AUP_Aed_proc(handle, &aedInputData, &aedOutputData);
  if (ret == 0) {
    *out_probability = aedOutputData.voiceProb;
    *out_flag = aedOutputData.vadRes;
  }
  return ret;
}

int ten_vad_destroy(ten_vad_handle_t* handle) {
  return AUP_Aed_destroy(handle);
}

const char* ten_vad_get_version(void) { return "1.0"; }