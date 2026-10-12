/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAPPACK_MPH_H_
#define _FA_MULTIMAPPACK_MPH_H_

#include "FAConfig.h"
#include "FAMap_judy.h"
#include "FASortMultiMap.h"
#include "FAChains2MinDfa_sort.h"
#include "FARSDfa2PerfHash.h"
#include "FAState2Ows_ar_uniq.h"
#include "FADfaPack_triv.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAMultiMapA;
class FAAllocatorA;

///
/// This class builds memory dump representation for the Multi Map refered
/// by some automaton or other structure. The class returns mapping from old
/// keys to new keys so the referee structure keys must be updated.
///
/// Note: 
/// 1. Such a representation is very efficient from the memory point 
/// of view for maps with huge number of keys as it uses RS-DFA to keep arrays 
/// of values.
///
/// 2. The memory dump representation is as follows:
///
/// BEGIN
/// <MaxArraySize>              : int
/// <Direction>                 : int
/// <DFA dump>                  : see FADfaPack_triv.h for details
/// END
///

class FAMultiMapPack_mph {

public:
    FAMultiMapPack_mph (FAAllocatorA * pAlloc);

public:
    /// sets up input map
    void SetMultiMap (const FAMultiMapA * pMMap);
    /// sets up direction, e.g. l2r, r2l in which to store multi-map arrays
    void SetDirection (const int Direction);
    /// builds dump
    void Process ();
    /// returns Old2New map for multi-map keys
    const FAMapA * GetOld2New () const;
    /// returns built dump
    const int GetDump (const unsigned char ** ppDump) const;

private:
    // returns object into initial state
    inline void Clear ();
    // helper method
    inline const int GetChain (const int Key, const int ** ppChain);
    inline void ProcessChains ();

private:
    // input multi-map
    const FAMultiMapA * m_pMMap;
    // direction to process arrays
    int m_Direction;
    // sorts mmap's arrays in specified order
    FASortMultiMap m_mmap_sort;
    // builds min dfa from the list of sorted arrays
    FAChains2MinDfa_sort m_arr2dfa;
    // builds MPH for the given min DFA
    FARSDfa2PerfHash m_dfa2mph;
    // keeps MPH's transition reactions
    FAState2Ows_ar_uniq m_state2ows;
    // packs min Moore-Multi DFA into memory dump
    FADfaPack_triv m_fsm2dump;
    // keeps OldKey -> NewKey map
    FAMap_judy m_old2new;
    // keeps resulting memory dump
    FAArray_cont_t < unsigned char > m_dump;
    // keeps current chain if needed
    FAArray_cont_t < int > m_chain_buff;
    int * m_pChainBuff;
    int m_MaxChainSize;

};

}

#endif
