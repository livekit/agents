//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __BIQUAD_H__
#define __BIQUAD_H__

#include <stdio.h>

#define AGORA_UAP_BIQUAD_MAX_SECTION (20)
// the max. number of sections supported by this Biquad module

#define AGORA_UAP_BIQUAD_MAX_INPUT_LEN (3840)
// max. number of samples each time can be fed in

#define AGORA_UAP_BIQUAD_ALIGN8(o) (((o) + 7) & (~7))
#define _BIQUAD_FLOAT2SHORT(x) \
  ((x) < -32767.5f ? -32768 : ((x) > 32766.5f ? 32767 : (short)floor(.5 + (x))))

#define _BIQUAD_DC_REMOVAL_NSECT (2)
const float _BIQUAD_DC_REMOVAL_B[_BIQUAD_DC_REMOVAL_NSECT][3] = {
    {1.0f, -2.0f, 1.0f}, {1.0f, -1.0f, 0.0f}};
const float _BIQUAD_DC_REMOVAL_A[_BIQUAD_DC_REMOVAL_NSECT][3] = {
    {1.0f, -1.93944294f, 0.94281253f}, {1.0f, -0.94276431f, 0.0f}};
// const float _BIQUAD_DC_REMOVAL_G[_BIQUAD_DC_REMOVAL_NSECT] = {0.97056387f,
// 0.97138215f};
const float _BIQUAD_DC_REMOVAL_G[_BIQUAD_DC_REMOVAL_NSECT] = {0.97056387f,
                                                              0.8655014957f};

// Configuration Parameters, which impacts dynamic memory occupation, can only
// be set during allocation
typedef struct Biquad_StaticCfg_ {
  size_t maxNSample;  // max. number of samples each time can be fed in
  // (0, AGORA_UAP_BIQUAD_MAX_INPUT_LEN]

  int nsect;  // the number of sections to be processed by this Biquad module
  // (-inf, AGORA_UAP_BIQUAD_MAX_SECTION]
  // if <= 0, use internal default filter coefficients

  const float* B[AGORA_UAP_BIQUAD_MAX_SECTION];
  const float* A[AGORA_UAP_BIQUAD_MAX_SECTION];
  // always assume A[...][0] = 1.0f
  const float* G;
} Biquad_StaticCfg;

typedef struct Biquad_InputData_ {
  const void*
      samplesPtr;  // externally provided buffer containing input time samples
  // either in short or float type
  short sampleType;  // = 0: samplesPtr = short*; o.w. samplesPtr = float*
  size_t nsamples;   // number of samples fed in this time
} Biquad_InputData;

typedef struct Biquad_OutputData_ {
  void* outputBuff;  // externally provided output buffer,
                     // assumed to be of enough size  nsamples *
                     // sizeof(short)/sizeof(short) output data type is the same
                     // as input
} Biquad_OutputData;

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************
 * AUP_Biquad_create(...)
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
int AUP_Biquad_create(void** stPtr);

/****************************************************************************
 * AUP_Biquad_destroy(...)
 *
 * destroy biquad instance, and releasing all the dynamically allocated memory
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
int AUP_Biquad_destroy(void** stPtr);

/****************************************************************************
 * AUP_Biquad_memAllocate(...)
 *
 * This function sets Static Config params and does memory allocation
 * operation
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
int AUP_Biquad_memAllocate(void* stPtr, const Biquad_StaticCfg* pCfg);

/****************************************************************************
 * AUP_Biquad_init(...)
 *
 * This function resets (initialize) the biquad module and gets it prepared for
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
int AUP_Biquad_init(void* stPtr);

/****************************************************************************
 * AUP_Biquad_getStaticCfg(...)
 *
 * This function get static configuration status from Biquad module
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
int AUP_Biquad_getStaticCfg(const void* stPtr, Biquad_StaticCfg* pCfg);

/****************************************************************************
 * AUP_Biquad_getAlgDelay(...)
 *
 * This function get algorithm delay from biquad module
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *
 * Output:
 *      - delayInSamples   : algorithm delay in terms of samples
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Biquad_getAlgDelay(const void* stPtr, int* delayInSamples);

/****************************************************************************
 * AUP_Biquad_proc(...)
 *
 * process a single frame
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *      - pCtrl         : per-frame variable control parameters
 *      - pIn           : input data stream
 *
 * Output:
 *      - pOut          : output data (mask, highband time-domain gain etc.)
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Biquad_proc(void* stPtr, const Biquad_InputData* pIn,
                    Biquad_OutputData* pOut);

#ifdef __cplusplus
}
#endif
#endif  // __BIQUAD_H__
