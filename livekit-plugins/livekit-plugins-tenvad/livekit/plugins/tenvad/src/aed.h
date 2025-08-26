//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __AED_H__
#define __AED_H__

#include <stdint.h>
#include <stdlib.h>

#define AUP_AED_MAX_FFT_SZ (1024)  // the max. fft-size supported by VAD module
#define AUP_AED_MAX_NBINS ((AUP_AED_MAX_FFT_SZ >> 1) + 1)

#define AUP_AED_FS (16000)  // assumed input freq.

// Configuration Parameters, which impacts dynamic memory occupation, can only
// be set during allocation
typedef struct Aed_StaticCfg_ {
  int enableFlag;  // flag to enable or disable this module
  // 0: disable, o.w.: enable
  size_t fftSz;               // fft-size, only support: 128, 256, 512, 1024
  size_t hopSz;               // fft-Hop Size, will be used to check
  size_t anaWindowSz;         // fft-window Size, will be used to calc rms
  int frqInputAvailableFlag;  // whether Aed_InputData will contain external
                              // freq. power-sepctra
} Aed_StaticCfg;

// Configuraiton parameters which can be modified/set every frames
typedef struct Aed_DynamCfg_ {
  float extVoiceThr;        // threshold for ai based voice decision [0,1]
  float extMusicThr;        // threshold for ai based music decision [0,1]
  float extEnergyThr;       // threshold for energy based vad decision [0, ---]
  size_t resetFrameNum;     // frame number for aivad reset [1875, 75000]
  float pitchEstVoicedThr;  // threshold for pitch-estimator to output estimated
                            // pitch
} Aed_DynamCfg;

// Spectrum are assumed to be generated with time-domain samples in [-32768,
// 32767] with or without pre-emphasis operation
typedef struct Aed_InputData_ {
  const float* binPower;  // [NBins], power spectrum of 16KHz samples
  int nBins;
  const float*
      timeSignal;  // [hopSz]   // this frame's input signal, in [-32768, 32767]
  int hopSz;       // should be equal to StaticCfg->hopSz
} Aed_InputData;

// return data from statistical ns module
typedef struct Aed_OutputData_ {
  float frameEnergy;  // frame energy for input normalized data
  float frameRms;     // rms for input int16 data
  int energyVadRes;  // vad res 0/1 with extEnergyThreshold based on input frame
                     // energy
  float voiceProb;   // vad score [0,1]
  int vadRes;  // vad res 0/1 with extVoiceThr based on ai method, t + 16ms res
               // correspond to the t input
  float pitchFreq;  // estimated pitch freq.
} Aed_OutputData;

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************
 * AUP_Aed_Create(...)
 *
 * This function creats a state handler from nothing, which is NOT ready for
 * processing
 *
 * Input:
 *
 * Output:
 *      - stPtr         : buffer to store the returned state handler
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_create(void** stPtr);

/****************************************************************************
 * AUP_Aed_Destroy(...)
 *
 * destroy VAD instance, and releasing all the dynamically allocated memory
 * this interface will also release ainsFactory, which was
 * created externally and passed to VAD module through memAllocate interface
 *
 * Input:
 *      - stPtr         : buffer of State Handler, after this method, this
 *                        handler won't be usable anymore
 *
 * Output:
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_destroy(void** stPtr);

/****************************************************************************
 * AUP_Aed_MemAllocate(...)
 *
 * This function sets Static Config params and does memory allocation
 * operation, will lose the dynamCfg values
 *
 * Input:
 *      - stPtr         : State Handler which was returned by _create
 *      - pCfg          : static configuration parameters
 *
 * Output:
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_memAllocate(void* stPtr, const Aed_StaticCfg* pCfg);

/****************************************************************************
 * AUP_Aed_init(...)
 *
 * This function resets (initialize) the VAD module and gets it prepared for
 * processing
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *
 * Output:
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_init(void* stPtr);

/****************************************************************************
 * AUP_Aed_setDynamCfg(...)
 *
 * This function set dynamic (per-frame variable) configuration
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *      - pCfg          : configuration content
 *
 * Output:
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_setDynamCfg(void* stPtr, const Aed_DynamCfg* pCfg);

/****************************************************************************
 * AUP_Aed_getStaticCfg(...)
 *
 * This function get static configuration status from VAD module
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *
 * Output:
 *      - pCfg          : configuration content
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_getStaticCfg(const void* stPtr, Aed_StaticCfg* pCfg);

/****************************************************************************
 * AUP_Aed_getDynamCfg(...)
 *
 * This function get dynamic (per-frame variable) configuration status from
 * VAD module
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *
 * Output:
 *      - pCfg          : configuration content
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_getDynamCfg(const void* stPtr, Aed_DynamCfg* pCfg);

/****************************************************************************
 * AUP_Aed_getAlgDelay(...)
 *
 * This function get algorithm delay from VAD module
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *
 * Output:
 *      - delayInFrms   : algorithm delay in terms of frames
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_getAlgDelay(const void* stPtr, int* delayInFrms);

/****************************************************************************
 * AUP_Aed_proc(...)
 *
 * process a single frame
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate and reset
 *      - pCtrl         : per-frame variable control parameters
 *      - pIn           : input data stream
 *
 * Output:
 *      - pOut          : output data (mask, highband time-domain gain etc.)
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Aed_proc(void* stPtr, const Aed_InputData* pIn, Aed_OutputData* pOut);

#ifdef __cplusplus
}
#endif

#endif
