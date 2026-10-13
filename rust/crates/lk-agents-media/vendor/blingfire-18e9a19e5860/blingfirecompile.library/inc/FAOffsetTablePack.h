/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_OFFSETTABLEPACK_H_
#define _FA_OFFSETTABLEPACK_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Packs offset tables, e.g. a growing array of non-negative numbers.
///
/// Current implementation works approximatelly as follows:
///   Offset = m_base [Idx >> m_SkipValue] + m_delta [Idx].
///

class FAOffsetTablePack {

public:
    FAOffsetTablePack (FAAllocatorA * pAlloc);

public:
    // sets up an array of offsets
    void SetOffsets (const unsigned int * pOffsets, const int OffsetCount);
    // builds memory dump
    void Process ();
    // returns built dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    // returns object into the initial state
    void Clear ();
    // calculates SizeOfBase
    inline const int CalcSizeOfBase () const;
    // calculates SkipValue
    inline const int CalcSkipValue () const;
    // stores offsets as a single base array
    void StoreUncompressed ();
    // calculates m_base and m_delta arrays
    void BuildArrays ();
    // stores offsets as a pair base + deltas
    void StoreCompressed ();
    // encodes offset
    void EncodeOffset (const int Offset);

private:
    // input offsets
    const unsigned int * m_pOffsets;
    int m_OffsetCount;
    // array of bases
    FAArray_cont_t < unsigned int > m_base;
    // array of deltas
    FAArray_cont_t < unsigned char > m_delta;
    // number of bytes to encode one base
    int m_SizeOfBase;
    // number of bits to cut to get index in m_base
    int m_SkipValue;
    // the resulting dump
    FAArray_cont_t < unsigned char > m_dump;
    unsigned char * m_pDump;
    unsigned int m_Offset;
};

}

#endif
