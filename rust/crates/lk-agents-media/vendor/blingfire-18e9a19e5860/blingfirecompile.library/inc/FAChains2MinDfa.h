/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_CHAINS2MINDFA_H_
#define _FA_CHAINS2MINDFA_H_

#include "FAConfig.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FAChains2MinDfa_sort.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// This class builds minimal RS DFA from the given list chains, the chains
/// don't have to be lexicographically sorted. 
///
/// Usage:
///  1. AddChain - for each chain
///  2. Process  - interface pointer cannot be used
///  3. GetDfa   - interface can be used
///
/// Note: This class should not be used for memory expensive chain lists 
/// as it keeps all chains in memory in explicit form.
///

class FAChains2MinDfa {

public:
    FAChains2MinDfa (FAAllocatorA * pAlloc);

public:
    /// adds chains
    void AddChain (const int * pChain, const int Size);
    /// makes the processing
    void Process ();
    /// returns RS DFA interface
    const FARSDfaA * GetRsDfa () const;

private:

    // compares chains in lexicographical order
    class _TChainCmp {
    public:
        _TChainCmp (FAArray_t < int > * pD);

    public:
        const bool operator () (const int i1, const int i2) const;

    private:
        FAArray_t < int > * m_pD;
    };

private:
    // all the chains in the form of [N, A_1, ..., A_N]
    FAArray_t < int > m_data;
    // the array of beginnings points of chains in m_data
    FAArray_cont_t < int > m_b;
    // temporary single chain storage
    FAArray_cont_t < int > m_tmp;
    // builds Min DFA from sorted chains
    FAChains2MinDfa_sort m_chains2dfa;

};

}

#endif
