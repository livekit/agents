/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CALCMEALY1_H_
#define _FA_CALCMEALY1_H_

#include "FAConfig.h"
#include "FARSNfa_wo_ro.h"
#include "FARSDfa_wo_ro.h"
#include "FAMealyDfa.h"
#include "FANfa2Dfa_t.h"
#include "FAArray_cont_t.h"
#include "FAMultiMap_judy.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;

///
/// This class is a helper for the FAMealyNfa2Dfa converter, it builds 
/// the first component of the decomposition. Don't use this class directly.
///

class FACalcMealy1 {

public:
    FACalcMealy1 (FAAllocatorA * pAlloc);

public:
    /// sets up input RS NFA
    void SetInNfa (const FARSNfaA * pInNfa);
    /// sets up state -> color map
    void SetColorMap (const int * pS2C, const int Size);
    /// makes conversion
    void Process ();
    /// returns object into the initial state
    void Clear ();

    /// returns output automaton
    const FARSDfaA * GetDfa () const;
    const FAMealyDfaA * GetSigma () const;

private:
    void Prepare ();

    inline const int Remap (
            const int * pStates, 
            const int Count, 
            const int ** ppStates
        );

    void RemapNfa ();

private:
    // input RS NFA
    const FARSNfaA * m_pInNfa;
    // color map
    const int * m_pS2C;
    int m_MapSize;
    int m_MaxColor;
    // reverse mapping C -> {S}
    FAMultiMap_judy m_c2s;
    // input (remapped) automaton
    FARSNfa_wo_ro m_in_nfa;
    // output Mealy DFA
    FARSDfa_wo_ro m_OutDfa;
    FAMealyDfa m_OutSigma;
    // determinization algorithm
    FANfa2Dfa_t < FARSNfa_wo_ro, FARSDfa_wo_ro > m_nfa2dfa;
    // temporary arrays for remapping destination states
    FAArray_cont_t < int > m_tmp;
    FAArray_cont_t < int > m_tmp2;
    FAArray_cont_t < int > m_alphabet;
};

}

#endif
