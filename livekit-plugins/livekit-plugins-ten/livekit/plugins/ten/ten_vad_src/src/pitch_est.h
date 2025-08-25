 //
 // Copyright © 2025 Agora
 // This file is part of TEN Framework, an open source project.
 // Licensed under the Apache License, Version 2.0, with certain conditions.
 //
 // Refer to the "LICENSE" file in the root directory for more information.
 //

#ifndef __PITCH_EST_H__
#define __PITCH_EST_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define AUP_PE_MAX_FFTSIZE (1024)
#define AUP_PE_MAX_NBINS ((AUP_PE_MAX_FFTSIZE >> 1) + 1)

#define AUP_PE_FS (16000)
// assumed sampling freq. of this module

// Configuration Parameters, which impacts dynamic memory occupation, can only
// be set during allocation
typedef struct PE_StaticCfg_ {
  size_t fftSz;        // fft-size, only support: 128, 256, 512, 1024
  size_t anaWindowSz;  // analysis fft-Window Size, will be used in LPC estimate
  size_t hopSz;        // fft-Hop Size, will be used to check
  int useLPCPreFiltering;
  // 0: use raw pcm to estimate pitch
  // 1: use LPC prefiltering before pitch estimation
  size_t procFs;  // internal processing sampling rate
  // 2000/4000/8000/16000
} PE_StaticCfg;

// Configuraiton parameters which can be modified/set every frames
typedef struct PE_DynamCfg_ {
  float voicedThr;  // threshold on frame correlation coeff to label if voice
                    // present
  // suggested value: procFs == 2KHz, Yes: 0.45, No: 0.4
} PE_DynamCfg;

// Spectrum are assumed to be generated with time-domain samples in [-32768,
// 32767] WITH LEC blowup protection Note: the input timeSignal has to be in
// 16KHz sampling-rate
typedef struct PE_InputData_ {
  const float*
      timeSignal;  // [hopSz]   // this frame's input signal, in [-32768, 32767]
  int hopSz;       // should be equal to StaticCfg->hopSz

  // if useLPCPreFiltering == 0, the following two input argument
  //    are not necessary
  const float* inBinPow;  // [nBins], bin-wise power
  int nBins;
} PE_InputData;

typedef struct PE_OutputData_ {
  float pitchFreq;  // the current estimated pitch freq.
  // <= 0: no voice
  int voiced;  // 0: no-voice, 1: voice-present
} PE_OutputData;

typedef struct PE_GetData_ {
  float pitchFreq;  // the current estimated pitch freq.
  // <= 0: no voice
  int voiced;  // 0: no-voice, 1: voice-present
} PE_GetData;

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************
 * AUP_PE_create(...)
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
int AUP_PE_create(void** stPtr);

/****************************************************************************
 * AUP_PE_destroy(...)
 *
 * destroy PE instance, and releasing all the dynamically allocated memory
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
int AUP_PE_destroy(void** stPtr);

/****************************************************************************
 * AUP_PE_memAllocate(...)
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
int AUP_PE_memAllocate(void* stPtr, const PE_StaticCfg* pCfg);

/****************************************************************************
 * AUP_PE_init(...)
 *
 * This function resets (initialize) the PE module and gets it prepared for
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
int AUP_PE_init(void* stPtr);

/****************************************************************************
 * AUP_PE_setDynamCfg(...)
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
int AUP_PE_setDynamCfg(void* stPtr, const PE_DynamCfg* pCfg);

/****************************************************************************
 * AUP_PE_getStaticCfg(...)
 *
 * This function get static configuration status from PE module
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
int AUP_PE_getStaticCfg(const void* stPtr, PE_StaticCfg* pCfg);

/****************************************************************************
 * AUP_PE_getDynamCfg(...)
 *
 * This function get dynamic (per-frame variable) configuration status from
 * PE module
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
int AUP_PE_getDynamCfg(const void* stPtr, PE_DynamCfg* pCfg);

/****************************************************************************
 * AUP_PE_getAlgDelay(...)
 *
 * This function get algorithm delay from PE module
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
int AUP_PE_getAlgDelay(const void* stPtr, int* delayInFrms);

/****************************************************************************
 * AUP_PE_proc(...)
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
int AUP_PE_proc(void* stPtr, const PE_InputData* pIn, PE_OutputData* pOut);

#ifdef __cplusplus
}
#endif
#endif  // __PITCH_EST_H__
