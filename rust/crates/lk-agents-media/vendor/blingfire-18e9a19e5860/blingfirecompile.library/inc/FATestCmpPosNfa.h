/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTCMPPOSNFA_H_
#define _FA_TESTCMPPOSNFA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaCA;
class FAMultiMapCA;

///
/// For the given two triplets: < FARSNfaCA, FAMultiMapCA, FAMultiMapCA > 
/// (FARSNfaCA - position NFA, FAMultiMapCA - mapping from states to the openning 
/// triangular brackets, FAMultiMapCA - mapping from states to the closing 
/// triangular brackets) identifies whether these triplets have the same behavior.
///
/// Note: States should have the meaning of positions in the corresponding regular
/// expression, e.g. should be enumerated in the same way for both automata.
///

class FATestCmpPosNfa {

public:
    FATestCmpPosNfa (FAAllocatorA * pAlloc);

public:
    // sets up first position NFA and mappings from state to triangular brackets
    // pState2BegTrBr1, pState2EndTrBr1 are optional
    void SetFsm1 (
            const FARSNfaCA * pNfa1, 
            const FAMultiMapCA * pState2BegTrBr1,
            const FAMultiMapCA * pState2EndTrBr1
    );
    // sets up second position NFA and mappings from state to triangular brackets
    // pState2BegTrBr2, pState2EndTrBr2 are optional
    void SetFsm2 (
            const FARSNfaCA * pNfa2, 
            const FAMultiMapCA * pState2BegTrBr2,
            const FAMultiMapCA * pState2EndTrBr2
    );
    // returns true if interfaces behaves identically
    const bool Process (const int MaxState, const int MaxIw);

private:
    const bool CheckMaxTrBrSize ();
    const bool CheckInitials () const;
    const bool CheckState2BegTrBr (const int State);
    const bool CheckState2EndTrBr (const int State);
    const bool CheckDest (const int State, const int Iw);

private:
    // input interfaces
    const FARSNfaCA * m_pNfa1;
    const FARSNfaCA * m_pNfa2;
    const FAMultiMapCA * m_pState2BegTrBr1;
    const FAMultiMapCA * m_pState2BegTrBr2;
    const FAMultiMapCA * m_pState2EndTrBr1;
    const FAMultiMapCA * m_pState2EndTrBr2;
    // temporary arrays
    FAArray_cont_t < int > m_tmp_arr1;
    FAArray_cont_t < int > m_tmp_arr2;
    int m_MaxTrBrBegSize;
    int m_MaxTrBrEndSize;
};

}

#endif
