/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CHAINSPACK_TRIV_H_
#define _FA_CHAINSPACK_TRIV_H_

#include "FAConfig.h"
#include "FAMap_judy.h"
#include "FAChain2Num_hash.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Packs chains of integers
/// 1. Uses byte, short or int for values encoding
/// 2. Stores more frequently used arrays of values by smaller offsets
///
/// Binary Format:
/// BEGIN
///   <SizeOfElement>                   : int
///   <Max array size>                  : int
///   <L_1><array_1>                    : SizeOfElement * (L_1 + 1)
///   <L_2><array_1>                    : SizeOfElement * (L_2 + 1)
///   ...
///   <L_N><array_N>                    : SizeOfElement * (L_N + 1)
/// END
///

class FAChainsPack_triv {

public:
    FAChainsPack_triv (FAAllocatorA * pAlloc);

public:
    // adds next array of [Values], you can add one array as many times
    // it is used, so it will have higher frequency value and lower offset
    void Add (const int * pValues, const int Count);
    // sets up min sizeof value, otherwise minimum possible is used
    void SetSizeOfValue (const int SizeOfValue);
    // calculates dump representation
    void Process ();
    // returns object into the initial state
    void Clear ();

public:
    // returns an offset of encoded [Values] array, or -1 if does not exist
    const int GetOffset (const int * pValues, const int Count) const;
    // returns memory dump pointer and size
    const int GetDump (const unsigned char ** ppDump) const;

private:
    void Prepare ();
    void EncodeValues (const int * pValues, const int Count);
    inline void EncodeValue (const int Value);

private:
    // set to frequency container
    FAChain2Num_hash m_set2freq;
    // mapping from < [Values] -> Freq > pair index to the dump offset
    FAMap_judy m_idx2offset;
    // keeps max Idx, number of sets
    int m_MaxIdx;
    // temporary arrays
    FAArray_cont_t < int > m_idx_by_freq;
    FAArray_cont_t < int > m_tmp_arr;
    // output memory dump
    FAArray_cont_t < unsigned char > m_dump;
    unsigned char * m_pDump;
    int m_LastOffset;
    // values size in bytes
    int m_SizeOfValue;
    // max set length in elements
    int m_MaxCount;
    // keeps total number of elements in all the sets plus length
    int m_ValuesCount;
};

}

#endif
