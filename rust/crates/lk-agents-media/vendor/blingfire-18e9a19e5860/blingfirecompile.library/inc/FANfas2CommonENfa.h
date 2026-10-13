/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFAS2COMMONENFA_H_
#define _FA_NFAS2COMMONENFA_H_

#include "FAConfig.h"
#include "FARSNfa_ar_judy.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;


///
/// Merges N-nfas together by putting e-transitions to the each one.
/// L(ENfa) = L(Nfa_1) U L(Nfa_2) ... U L(Nfa_N)
///

class FANfas2CommonENfa {

public:
    FANfas2CommonENfa (FAAllocatorA * pAlloc);

public:
    /// specifies whether to add Nfa num info or not, does not add by default
    void SetAddNfaNums (const bool AddNfaNums);
    /// specifies the base Iw for the NfaNum information, 0 is used by default
    void SetNfaNumBase (const int NfaNumBase);
    /// make the resulting automaton empty
    void Clear ();
    /// specifies the Epsilon symbol
    void SetEpsilonIw (const int EpsilonIw);
    /// adds one more nfa to the common resulting nfa
    void AddNfa (const FARSNfaA * pNfa);
    /// makes the result usable
    void Process ();
    /// returns common nfa
    const FARSNfaA * GetCommonNfa () const;

private:
    /// copies transitions into the common nfa making renumeration
    inline void AddTransitions (const int StateOffset);
    /// copies final states making renumeration
    inline void AddFinals (const int StateOffset);
    /// copies initial states making renumeration and adds epsilon transitions
    inline void AddInitials (const int StateOffset);
    /// adds Nfa Nums into the common Nfa
    inline void AddNfaNums ();

private:
    /// true if no automatons has been added yet
    bool m_empty;
    /// input nfa
    const FARSNfaA * m_pInNfa;
    /// epsilon transition
    int m_EpsilonIw;
    /// the resulting automaton
    FARSNfa_ar_judy m_common_nfa;
    /// temporary storage for states renumeration
    FAArray_cont_t < int > m_states;
    /// epsilon destination states
    FAArray_cont_t < int > m_eps_dst;
    /// 
    bool m_AddNfaNums;
    int m_NfaNumBase;
    /// mapping from the final state num into NfaNum
    FAArray_cont_t < int > m_fsnum2nfanum;
    /// number of nfas added
    int m_NfaCount;
};

}

#endif
