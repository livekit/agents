/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_UTF16UTILS_H_
#define _FA_UTF16UTILS_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

/// Converts UTF-16LE/BE symbol into int (UTF-32LE),
/// returns new pointer in the multi byte sequence,
/// NULL if the sequence is not a valid UTF-16LE/BE string
const wchar_t * FAUtf16ToInt (
        const wchar_t * ptr,
        int * result,
        bool fBE
    );


/// Converts int (UTF-32LE) into UTF-16LE/BE symbol
/// returns end pointer of the output sequence
wchar_t * FAIntToUtf16 (
        int Symbol, 
        __out_ecount(MaxSize) wchar_t * ptr, 
        const int MaxSize,
        bool fBE
    );


/// Converts UTF-16LE/BE string of specified length to the array of ints.
/// Returns the number of used elements in the array.
/// Returns -1 if the input sequence is invalid.
const int FAStrUtf16ToArray (
        const wchar_t * pStr,
        const int Len,
        __out_ecount(MaxSize) int * pArray,
        const int MaxSize,
        bool fBE
    );


/// Converts array of ints (UTF-32LE) into UTF-16LE/BE string of  upto MaxStrSize
/// length, does not place terminating 0-byte. Returns output string length.
/// Returns -1 for invalid input sequence.
const int FAArrayToStrUtf16 (
        const int * pArray, 
        const int Size, 
        __out_ecount(MaxStrSize) wchar_t * pStr, 
        const int MaxStrSize,
        bool fBE
    );

/// returns true if pEncName is a valid UTF-16LE encoding name
const bool FAIsUtf16LeEnc (const char * pEncName);

/// returns true if pEncName is a valid UTF-16BE encoding name
const bool FAIsUtf16BeEnc (const char * pEncName);

}

#endif
