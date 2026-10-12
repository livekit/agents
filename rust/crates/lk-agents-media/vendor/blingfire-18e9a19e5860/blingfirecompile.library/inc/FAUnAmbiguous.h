/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_UNAMBIGUOUS_H_
#define _FA_UNAMBIGUOUS_H_

#include "FAConfig.h"
#include "FAChain2Num_hash.h"
#include "FAArray_cont_t.h"
#include "FARSNfa_ar_judy.h"
#include "FAMealyNfa.h"
#include "FARemoveUnreachable.h"

namespace BlingFire
{

class FAAllocatorA;
class FALessA;

///
/// This processor takes ambiguous Mealy NFA and returns unambiguous Mealy NFA
///
/// For example:
///
/// Input NFA(S(('a:1''b:1')|(('a:2'|'p:2')'b:2'))S)
/// S a b S -> S 1 1 S
/// S a b S -> S 2 2 S
/// S p b S -> S 2 2 S
///
/// Output NFA
/// S a b S -> S 1 1 S
/// S p b S -> S 2 2 S
///
/// Note: This algorithm is described in MERL TR-96-13
///

class FAUnAmbiguous {

public:
    FAUnAmbiguous (FAAllocatorA * pAlloc);

public:
    /// sets up comparison operator, if not set then < operator is used
    void SetLess (const FALessA * pLess);
    /// sets up input automaton
    void SetInMealy (const FARSNfaA * pInNfa, const FAMealyNfaA * pInSigma);
    /// makes processing
    void Process ();
    /// returns output automaton
    const FARSNfaA * GetNfa () const;
    const FAMealyNfaA * GetSigma () const;
    /// returns object into the initial state
    void Clear ();

private:
    void Prepare ();
    inline const int AddState (const int * pXS, const int Size);
    inline const int GetXS (const int Q, const int ** ppXS) const;
    inline const bool IsInitial (const int State) const;
    inline const bool IsFinal (const int State) const;
    // returns false if this new state should not be added
    const bool CalcNewState (
            const int X1,
            const int Iw,
            const int Y1,
            const int Ow1,
            const int * pS, 
            const int SSize,
            const int * pDst, 
            const int Count
        );
    void PrepareOutNfa ();
    void CreateOutNfa ();
    void RmUnreach ();

private:
    /// input
    const FALessA * m_pLess;
    const FARSNfaA * m_pInNfa;
    const FAMealyNfaA * m_pInSigma;
    /// output
    FARSNfa_ar_judy m_OutNfa;
    FAMealyNfa m_OutSigma;
    /// keeps bi-directional mapping Q <--> < X, {S} >, where Q is a new state,
    /// X is an old state and {S} is a set of states with a strictly less path
    FAChain2Num_hash m_q2xs;
    /// max new state stored in m_q2xs
    int m_MaxQ;
    /// temporary array
    FAArray_cont_t < int > m_xs;
    FAArray_cont_t < int > m_initials;
    FAArray_cont_t < int > m_finals;
    /// removes unreachable states
    FARemoveUnreachable m_rm;
};

}

#endif
