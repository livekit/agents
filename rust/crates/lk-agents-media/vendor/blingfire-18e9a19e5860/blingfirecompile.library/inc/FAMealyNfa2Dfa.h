/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYNFA2DFA_H_
#define _FA_MEALYNFA2DFA_H_

#include "FAConfig.h"
#include "FANfa2EqPairs.h"
#include "FAColorGraph_t.h"
#include "FAArray_cont_t.h"
#include "FARSNfa2RevNfa.h"
#include "FARSDfa_wo_ro.h"
#include "FARSNfa_wo_ro.h"
#include "FARSDfa_ro.h"
#include "FASplitStates.h"
#include "FACalcMealy1.h"
#include "FACalcMealy2.h"
#include "FAMealyDfa.h"

namespace BlingFire
{

class FAAllocatorA;
class FAMealyNfaA;

///
/// This processor takes Mealy NFA A with minimal deterministic projection
/// over Iw:Ow and if A has non-deterministic projection over Iw the algorithm
/// constructs the equivalent cascade of left-to-right minimal deterministic 
/// Mealy DFA A1 and right-to-left Mealy NFA A2. However, if the relation
/// representeted by A is functional then A2 is also a minimal deterministic
/// Mealy DFA.
///
/// Notes: 
///
/// 1. It is not necessary to provide minimal deterministic projection 
/// over Iw:Ow however the output relation won't be minimal if not to do so.
///
/// 2. If it is enough to have only one left-to-right automaton to represent 
/// the given relation then the algorithm won't construct trivial right-to-left
/// automaton.
///

class FAMealyNfa2Dfa {

public:
    FAMealyNfa2Dfa (FAAllocatorA * pAlloc);

public:
    /// sets up whether A1 and A2 should work as bi-machine, false by default
    void SetUseBiMachine (const bool UseBiMachine);
    /// sets up input mealy automaton
    void SetInNfa (
            const FARSNfaA * pInNfa, 
            const FAMealyNfaA * pInSigma
        );
    /// sets up output containers for Fsm1 (left-to-right)
    void SetOutFsm1 (FARSDfaA * pFsm1Dfa, FAMealyDfaA * pFsm1Ows);
    /// sets up output containers for optional Fsm2 (right-to-left)
    void SetOutFsm2 (FARSDfaA * pFsm2Dfa, FAMealyDfaA * pFsm2Ows);
    /// makes conversion
    void Process ();
    /// returns object into the initial state and frees memory
    void Clear ();

public:
    /// returns true if the resulting second automaton is non-deterministic
    const bool IsNonDet () const;
    /// returns second non-deterministic automaton
    const FARSNfaA * GetNfa2 () const;
    /// returns second non-deterministic sigma function
    const FAMealyNfaA * GetSigma2 () const;

private:
    void CreateDfa_triv ();
    void CalcEqPairs ();
    void ColorGraph ();
    void SplitStates ();
    void CalcRevNfa ();
    void CalcFsm1 ();
    void CalcFsm2 ();
    void RmArcs1 ();
    void SetMaxClasses ();

private:
    class _TEqGraph {
    public:
        _TEqGraph (const int * pV, const int Count, const FANfa2EqPairs * pE);

    public:
        const int GetVertices (const int ** ppV) const;
        const int GetArcCount () const;
        void GetArc (const int Num, int * pFrom, int * pTo) const;

    private:
        const int * const m_pV;
        const int m_Count;
        const FANfa2EqPairs * const m_pE;
    };

private:
    // input Mealy NFA
    const FARSNfaA * m_pInNfa; 
    const FAMealyNfaA * m_pInOws;

    // output containers
    FAMealyDfaA * m_pFsm1Ows;
    FARSDfaA * m_pFsm1Dfa;
    FAMealyDfaA * m_pFsm2Ows;
    FARSDfaA * m_pFsm2Dfa;

    FAArray_cont_t < int > m_tmp;
    FANfa2EqPairs m_eq_pairs;
    FAColorGraph_t < _TEqGraph > m_q2c;
    FASplitStates m_split_states;
    FARSNfa2RevNfa m_rev;
    FARSNfa_wo_ro m_rev_nfa;
    FACalcMealy1 m_calc_fsm1;
    FACalcMealy2 m_calc_fsm2;

    bool m_UseBiMachine;
};

}

#endif
