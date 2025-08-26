//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __STFT_H__
#define __STFT_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define AUP_STFT_MAX_FFTSZ (4096)

// Configuration Parameters, which impacts dynamic memory occupation, can only
// be set during allocation
typedef struct Analyzer_StaticCfg_ {
  int win_len;
  int hop_size;
  int fft_size;
  const float* ana_win_coeff;
} Analyzer_StaticCfg;

// Spectrum are assumed to be generated with time-domain samples in [-32768,
// 32767] WITH LEC blowup protection Note: the input timeSignal has to be in
// 16KHz sampling-rate
typedef struct Analyzer_InputData_ {
  float* input;
  int iLength;
} Analyzer_InputData;

typedef struct Analyzer_OutputData_ {
  float* output;  // externally provided buffe
  int oLength;    // externally provided buffer length
} Analyzer_OutputData;

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************
 * AUP_Analyzer_create(...)
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
int AUP_Analyzer_create(void** stPtr);

/****************************************************************************
 * AUP_Analyzer_destroy(...)
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
int AUP_Analyzer_destroy(void** stPtr);

/****************************************************************************
 * AUP_Analyzer_memAllocate(...)
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
int AUP_Analyzer_memAllocate(void* stPtr, const Analyzer_StaticCfg* pCfg);

/****************************************************************************
 * AUP_Analyzer_init(...)
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
int AUP_Analyzer_init(void* stPtr);

/****************************************************************************
 * AUP_Analyzer_getStaticCfg(...)
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
int AUP_Analyzer_getStaticCfg(const void* stPtr, Analyzer_StaticCfg* pCfg);

/****************************************************************************
 * AUP_Analyzer_proc(...)
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
int AUP_Analyzer_proc(void* stPtr, const Analyzer_InputData* pIn,
                      Analyzer_OutputData* pOut);

#ifdef __cplusplus
}
#endif
#endif  // __STFT_H__
