/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STRINGARRAYPACK_H_
#define _FA_STRINGARRAYPACK_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Packs the array of arbitrary strings or byte sequences.
/// 
/// Dump format: 
///   <Count>        - int (keeps number of elements in the array N)
///   <offset1>      - int
///   <offset2>      - int
///   ...
///   <offsetN>      - int
///   <rawBytes>     - strings concatenated one after another
///

class FAStringArrayPack {

public:
    FAStringArrayPack (FAAllocatorA * pAlloc);

public:
    /// sets up the array of values
    void SetArray (const FAArray_cont_t < unsigned char > * pBuff, const FAArray_cont_t < int > * pOffsets);
    /// builds memory dump
    void Process ();
    /// returns built dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    const FAArray_cont_t < unsigned char > * m_pBuff;
    const FAArray_cont_t < int > * m_pOffsets;
    // the resulting dump
    FAArray_cont_t < unsigned char > m_dump;
};

}

#endif
