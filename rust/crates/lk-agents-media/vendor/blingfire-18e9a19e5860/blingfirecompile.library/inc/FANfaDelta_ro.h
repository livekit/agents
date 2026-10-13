/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_NFADELTA_RO_H_
#define _FA_NFADELTA_RO_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Constant Delta function of NFA.
///
/// Note: transitions has to be sorted first by State then by Iw!
///

class FANfaDelta_ro {

public:
    FANfaDelta_ro (FAAllocatorA * pAlloc);

public:
    /// adds transitions
    void AddTransition (
            const int State,
            const int Iw,
            const int * pDsts,
            const int Count
        );
    /// returns into the initial state and frees memory
    void Clear ();

public:
    // returns Iws for the given State
    const int GetIWs (
            const int State, 
            const int ** ppIws
        ) const;

    // returns Dsts for the given State, Iw pair
    const int GetDest (
            const int State, 
            const int Iw, 
            const int ** ppDsts
        ) const;

private:
    /// State --> <First, Count>, where First is index in m_IwDstIdx
    FAArray_cont_t < int > m_State2FC;

    /// two parallel arrays: m_Iws and 
    /// Iws: [m_Iws [First], ...,  m_Iws [First + Count]]
    FAArray_cont_t < int > m_Iws;

    /// Dsts: [m_Dsts [First], ...,  m_Dsts [First + Count]]
    /// if m_Dsts [i] >= 0 then
    ///   Dsts == { m_Dsts [i] }              // a set of one element
    /// else
    ///   Dsts == m_Sets [-(m_Dsts [i] + 1)]  // a set from m_Sets
    FAArray_cont_t < int > m_Dsts;

    /// DstIdx <--> DstSet map, keeps only sets of size > 1
    FAChain2Num_hash m_Sets;
};

}

#endif
