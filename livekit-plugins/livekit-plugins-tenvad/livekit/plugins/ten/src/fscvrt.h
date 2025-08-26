//
// Copyright © 2025 Agora
// This file is part of TEN Framework, an open source project.
// Licensed under the Apache License, Version 2.0, with certain conditions.
// Refer to the "LICENSE" file in the root directory for more information.
//
#ifndef __FSCVRT_H__
#define __FSCVRT_H__

#define AUP_FSCVRT_MAX_INPUT_LEN (2400)
// max. number of samples each time can be fed in

#include <stdio.h>

typedef struct FscvrtStaticCfg_ {
  int inputFs;     // input stream sampling freq.
  int outputFs;    // output stream sampling freq.
  int stepSz;      // number of input samples per each proc.
  int inputType;   // input data type, 0: short, 1: float
  int outputType;  // output data type, 0: short, 1: float
} FscvrtStaticCfg;

typedef struct FscvrtInData_ {
  const void* inDataSeq;  // [stepSz], externally provided buffer
  int outDataSeqLen;
  // the length of externally provided buffer outDataSeq in OutData
} FscvrtInData;

typedef struct FscvrtOutData_ {
  int nOutData;  // number of samples in outDataSeq
  // this value may vary by +-1 from frame-to-frame
  // and the user needs to check if nOutData <= outDataSeqLen
  // o.w. the buffer outDataSeq is not long enough
  int outDataType;   // output data type, 0: short, 1: float
  void* outDataSeq;  // [outDataSeqLen], externally provided buffer
} FscvrtOutData;

typedef struct FscvrtGetData_ {
  int maxOutputStepSz;  // max. number of output samples per each proc.
  int delayInInputFs;   // algorithm delay in terms of samples @ input fs
} FscvrtGetData;

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************
 * AUP_Fscvrt_create(...)
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
int AUP_Fscvrt_create(void** stPtr);

/****************************************************************************
 * AUP_Fscvrt_destroy(...)
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
int AUP_Fscvrt_destroy(void** stPtr);

/****************************************************************************
 * AUP_Fscvrt_memAllocate(...)
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
int AUP_Fscvrt_memAllocate(void* stPtr, const FscvrtStaticCfg* pCfg);

/****************************************************************************
 * AUP_Fscvrt_init(...)
 *
 * This function resets (initialize) the XXXX module and gets it prepared for
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
int AUP_Fscvrt_init(void* stPtr);

/****************************************************************************
 * AUP_Fscvrt_setDynamCfg(...)
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
int AUP_Fscvrt_setDynamCfg(void* stPtr);

/****************************************************************************
 * AUP_Fscvrt_getStaticCfg(...)
 *
 * This function get static configuration status from XXXXX module
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
int AUP_Fscvrt_getStaticCfg(const void* stPtr, FscvrtStaticCfg* pCfg);

/****************************************************************************
 * AUP_Fscvrt_getInfor(...)
 *
 * This function get subsidiary information from Fs-Converter module
 *
 * Input:
 *      - stPtr         : State Handler which has gone through create and
 *                        memAllocate
 *
 * Output:
 *      - FscvrtGetData : returned information
 *
 * Return value         :  0 - Ok
 *                        -1 - Error
 */
int AUP_Fscvrt_getInfor(const void* stPtr, FscvrtGetData* buff);

/****************************************************************************
 * AUP_Fscvrt_proc(...)
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
int AUP_Fscvrt_proc(void* stPtr, const FscvrtInData* pIn, FscvrtOutData* pOut);

#ifdef __cplusplus
}
#endif
#endif  // __FSCVRT_H__
