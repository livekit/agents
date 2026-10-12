/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFAS2TUPLENFA_H_
#define _FA_NFAS2TUPLENFA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_judy.h"
#include "FAEncoder_pref.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;

///
/// This class build tuple-NFA from the array of N parallel Nfa-s.
///
/// Nfa_a and Nfa_b are parallel if the following holds:
///  |Qa| == |Qb| and if qa == qb then Pa == Pb,
///   where \forall pa \in Pa, <qa, iwa, pa> \in Ea, iwa \in IWa
///   and \forall pb \in Pb, <qb, iwb, pb> \in Eb, iwb \in IWb.
///
/// The transitions <q, <Iw_1, ..., Iw_N>, p> of tuple-NFA where
/// <qi, Iw_i, pi> \in Nfa_i and qi == q, pi == p are represented as N
/// consiquent transitions.
///
/// Note:
///
/// 1. This class does not copy input NFAs (just store the pointers) so NFAs
///    must exist during the entire processing.
///

class FANfas2TupleNfa {

public:
    FANfas2TupleNfa (FAAllocatorA * pAlloc);

public:
    /// sets up ignore interval
    void SetIgnoreBase (const int IgnoreBase);
    /// sets up ignore interval
    void SetIgnoreMax (const int IgnoreMax);
    /// adds one more NFA to merge
    void AddNfa (const FARSNfaA * pNfa);
    /// sets up output container
    void SetOutNfa (FARSNfaA * pNfa);
    /// makes transformation
    void Process ();
    /// returns object into the initial state
    void Clear ();

private:
    inline const int GetMaxState () const;
    inline const int GetMaxIw () const;
    inline const int GetInitials (const int ** ppStates) const;
    inline const int GetFinals (const int ** ppStates) const;

    inline void RemapStates (
            const int * pStates, 
            const int Count, 
            const int ** ppOutStates
        );
    inline void RemapStates2 (
            const int FromState,
            const int Idx,
            const int * pStates, 
            const int Count, 
            const int ** ppOutStates
        );

    inline const bool Ignore (const int Iw) const;

    void CalcStateMap ();
    void AddTransitions (const int Idx, const int State);
    void AddIgnoredTransitions (const int State);

private:
    // input
    FAArray_cont_t < const FARSNfaA * > m_NfaArr;
    const FARSNfaA ** m_pNfaArr;
    int m_Count;
    // output
    FARSNfaA * m_pOutNfa;
    // mapping from state / transition -> new state
    FAChain2Num_judy m_state2state;
    FAEncoder_pref m_encoder;
    // keeps Max new state number
    int m_NewMaxState;
    // temporary arrays for states remapping
    FAArray_cont_t < int > m_tmp;
    // ignore interval
    int m_IgnoreBase;
    int m_IgnoreMax;
};

}

#endif
