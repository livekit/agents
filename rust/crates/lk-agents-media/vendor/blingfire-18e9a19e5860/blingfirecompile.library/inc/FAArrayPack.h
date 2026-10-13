/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ARRAYPACK_H_
#define _FA_ARRAYPACK_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Packs the big array of arbitrary int values.
///
/// The array is split into chunks of M elements each. The unique chunks 
/// are stored in the Data array. The following condition holds:
///  Array [Idx] == Data [Index [Idx/M] + (Idx%M)]
///
/// if |Data| + |Index| > |Array| - C then array is stored as is -- flat.
/// 
/// Dump Header: 
///   <M>       - unsigned int  ; 1..8
///   <IdxSize> - unsigned int  ; 0 if M == 1
///   <ValSize> - unsigned int  ; 1..4
///   <Count>   - unsigned int
///

class FAArrayPack {

public:
    FAArrayPack (FAAllocatorA * pAlloc);

public:
    /// sets up the array of values
    void SetArray (const int * pArr, const int Size);   
    /// forces the array to be flat (to have one level), false by default
    void SetForceFlat (const bool ForceFlat);
    /// builds memory dump
    void Process ();
    /// returns built dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    /// returns the number of bytes necessary to store the Val
    static inline const int GetSizeOf (const int Val);

    /// calculates max value size
    inline const int CalcSizeOfVal () const;
    /// calculates the best M value [2..8]
    inline void CalcBestM ();
    /// adds chain into m_Data container
    inline const int AddChain (const int FromIdx, const int Count);

    // builds flat dump
    void BuildFlat ();
    // builds packed dump
    void BuildPacked ();

    /// returns object into the initial state
    void Clear ();

private:
    // the input array
    const int * m_pArr;
    int m_Size;
    // the resulting dump
    FAArray_cont_t < unsigned char > m_dump;
    // best M, m_M == 1 if flat representation is used
    int m_M;
    // chunk <-> id mapping
    FAChain2Num_hash m_Data;
    // temporary chain
    FAArray_cont_t < int > m_tmp_chain;
    // idx/M -> id
    FAArray_cont_t < int > m_Index;
    // bytes per value
    int m_SizeOfValue;
    // bytes per |m_Index [i]|
    int m_SizeOfIdx;
    /// m_M will always be 1
    bool m_ForceFlat;

    enum {
        // acceptable difference between packed and flat representations
        MinDiffSize = 10000,
    };
};

}

#endif
