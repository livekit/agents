/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CALCMEALY2_H_
#define _FA_CALCMEALY2_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FARSNfa_wo_ro.h"
#include "FARSNfa2RevNfa.h"
#include "FAMealyNfa.h"

#include <set>

namespace BlingFire
{

class FARSDfaA;
class FAMealyDfaA;

///
/// This class is a helper for the FAMealyNfa2Dfa converter, it builds 
/// the second component of the decomposition. Don't use this class directly.
///

class FACalcMealy2 {

public:
    FACalcMealy2 (FAAllocatorA * pAlloc);

public:
    /// sets up reversed RS NFA
    void SetInNfa (const FARSNfaA * pInRevNfa);
    /// sets up sigma function of the original automaton
    void SetSigma (const FAMealyNfaA * pOrigSigma);
    /// sets up first component of the cascade
    void SetMealy1 (const FARSDfaA * pInDfa, const FAMealyDfaA * pInSigma);
    /// calculates RS component of the Mealy NFA/DFA 2
    /// Note: Sigma function is left original, as output RS DFA has exactly 
    /// the same states as input Mealy NFA
    void Process ();
    /// returns RS component of the Mealy NFA/DFA 2
    const FARSNfaA * GetOutNfa () const;
    const FAMealyNfaA * GetSigma () const;
    /// returns object into the initial state
    void Clear ();

private:
    inline void Prepare ();
    inline void EnqueueInitials ();
    inline void SubstNfa ();
    inline void Reverse ();

private:
    const FARSNfaA * m_pInRevNfa;
    const FAMealyNfaA * m_pOrigSigma;
    const FARSDfaA * m_pInDfa;
    const FAMealyDfaA * m_pInSigma;

    // stack
    FAArray_cont_t < int > m_s;
    std::set < std::pair < int, int > > m_P;

    // reversed automaton with substituted inputs
    FARSNfa_wo_ro m_tmp_out;
    // reversing algorithm
    FARSNfa2RevNfa m_rev;
    // direct automaton with substituted inputs
    FARSNfa_wo_ro m_OutNfa;
    FAMealyNfa m_OutSigma;
};

}

#endif
