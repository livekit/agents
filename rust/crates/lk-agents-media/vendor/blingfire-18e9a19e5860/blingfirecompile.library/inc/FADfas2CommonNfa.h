/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DFAS2COMMONNFA_H_
#define _FA_DFAS2COMMONNFA_H_

#include "FAConfig.h"
#include "FARSNfa_wo_ro.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FARSDfaA;
class FAAllocatorA;

///
/// Merges N DFAs together without using e-removal.
/// L(Nfa) = L(Dfa_1) U L(Dfa_2) ... U L(Dfa_N)
///

class FADfas2CommonNfa {

public:
    FADfas2CommonNfa (FAAllocatorA * pAlloc);

public:
    /// returns object into the initial state
    void Clear ();
    /// adds one more DFA to the common resulting nfa
    void AddDfa (const FARSDfaA * pDfa);
    /// makes the result usable
    void Process ();
    /// returns common nfa
    const FARSNfaA * GetCommonNfa () const;

private:
    /// checks whether old initial state is reachable or not
    inline const bool IsReachable (const FARSDfaA * pDfa) const;

    /// adds transitions from the DFA's state for the NFA's State
    inline void AddTransitions (
            const FARSDfaA * pDfa, 
            const int State, 
            const int state
        );

private:
    // common NFA
    FARSNfa_wo_ro m_nfa;
    // final states
    FAArray_cont_t < int > m_finals;
    // NFA's state count
    int m_StateCount;
    // m_D == -1 if DFA's initial state is not needed, 0 otherwise
    int m_D;
    // DFA's alphabet
    const int * m_pIws;
    // DFA's alphabet size
    int m_IwsCount;
};

}

#endif
