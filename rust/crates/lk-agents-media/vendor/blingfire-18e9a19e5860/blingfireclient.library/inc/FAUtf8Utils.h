/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_UTF8UTILS_H_
#define _FA_UTF8UTILS_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

/// Converts UTF-8 symbol into int (UTF-32LE),
/// returns new pointer in the multi byte sequence,
/// NULL if the sequence is not a valid UTF8 string
const char * FAUtf8ToInt (const char * ptr, int * result);

/// Converts UTF-8 symbol into int (UTF-32LE),
/// returns new pointer in the multi byte sequence,
/// NULL if the sequence is not a valid UTF8 string
/// empty input buffers are not allowed
const char * FAUtf8ToInt (
        const char * pBegin, 
        const char * pEnd, 
        int * pResult
    );

/// Converts int (UTF-32LE) into UTF-8 symbol
/// returns end pointer of the output sequence
char * FAIntToUtf8 (
        const int Symbol, 
        __out_ecount(MaxSize) char * ptr,
        const int MaxSize
    );

/// returns the size of UTF-8 symbol
/// returns 0 if it is not a valid symbol
const int FAUtf8Size (const char * ptr);

/// returns the size of UTF-32LE symbol in UTF-8 encoding
/// returns 0 if it is not a valid symbol
const int FAUtf8Size (const int Symbol);

/// Converts 0-terminated UTF8 string to the array of ints.
/// Returns the number of used elements in the array.
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf8ToArray (
        const char * pStr, 
        __out_ecount(MaxSize) int * pArray, 
        const int MaxSize
    );

/// Converts UTF8 string of specified length to the array of ints.
/// Returns the number of used elements in the array.
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf8ToArray (
        const char * pStr, 
        const int Len, 
        __out_ecount(MaxSize) int * pArray, 
        const int MaxSize
    );

/// Converts UTF8 string of specified length to the array of ints and
/// for each UTF-32 character returns its offset in the pStr.
/// Returns the number of used elements in the array.
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf8ToArray (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) int * pArray,
        __out_ecount(MaxSize) int * pOffsets,
        const int MaxSize
    );


/// Converts UTF8 string of specified length to the array of ints
///  using each byte as a character.
/// Returns the number of used elements in the array.
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf8AsBytesToArray (
        const char * pStr, 
        const int Len, 
        __out_ecount(MaxSize) int * pArray, 
        const int MaxSize
    );

/// Converts UTF8 string of specified length to the array of ints
///  using each byte as a character and for each returned character
//   returns its offset in the pStr.
/// Returns the number of used elements in the array.
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf8AsBytesToArray (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) int * pArray,
        __out_ecount(MaxSize) int * pOffsets,
        const int MaxSize
    );


/// Converts int (UTF-32LE) into UTF-16LE encoded string
/// returns end pointer of the output sequence
/// return NULL if buffer length is insufficient
wchar_t * FAIntToUtf16LE (
        int Symbol,
        __out_ecount(MaxSize) wchar_t * ptr,
        const int MaxSize
    );

/// Converts UTF-8 string of specified length to UTF-16LE string
/// for each UTF-16 character returns its offset in the pStr.
/// Returns the number of used elements in the array.
/// If the input buffer is not sufficient for all input sequences, decodes as many that fit in the supplied buffer
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf8ToUtf16LE (
        const char * pStr,
        const int Len,
        __out_ecount(MaxSize) wchar_t * pArray,
        __out_ecount(MaxSize) int * pOffsets,
        const int MaxSize
    );

/// Converts array of ints (UTF-32LE) into UTF-8 string of  upto MaxStrSize
/// length, does not place terminating 0-byte. Returns output string length.
/// Returns -1 for invalid input sequence.
const int FAArrayToStrUtf8 (
        const int * pArray, 
        const int Size, 
        __out_ecount(MaxStrSize) char * pStr, 
        const int MaxStrSize
    );

/// returns true if pEncName is a valid UTF-8 encoding name
const bool FAIsUtf8Enc (const char * pEncName);


/// Normalizes characters in the valid input UTF-8 string and returns a valid
/// UTF-8 string. if the input string is invalid UTF-8 then -1 is returned 
/// otherwise the output string size is returned. 
///
/// Note: if the output buffer is not big enough, the needed size is returned
///   but no actual data is copied to the output buffer.
///
/// Parameters:
///  pStr, Len represent the input string
///  pOutStr, MaxOutSize output buffer of MaxOutSize elements
///  NormAlgo specifies the normalization algorithm :
///   FAFsmConst::NORMALIZE_DEFAULT -- current production normalization for case and diacritics
///   FAFsmConst::NORMALIZE_PRESERVE_DIACRITICS -- preserves all diacritics, but drops the case
///   FAFsmConst::NORMALIZE_REMOVE_DIACRITICS -- removes diacritics (more than DEFAULT) and drops the case
///
const int FAStrUtf8Normalize (
        const char * pStr,
        const int Len,
        __out_ecount(MaxOutSize) char * pOutStr,
        const int MaxOutSize,
        const int NormAlgo
    );


/// UTF-8 constants
class FAUtf8Const {
public:
    enum {
        MAX_CHAR_SIZE = 4,
    };
};

}

#endif
