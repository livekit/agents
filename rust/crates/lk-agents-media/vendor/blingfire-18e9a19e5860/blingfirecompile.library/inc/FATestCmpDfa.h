/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTCMPDFA_H_
#define _FA_TESTCMPDFA_H_

#include "FAConfig.h"
#include "FAArray_t.h"
#include "FABitArray.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaCA;
class FAState2OwCA;
class FAState2OwsCA;
class FAMealyDfaCA;

///
/// For the given two triplets: < FARSDfaCA, FAState2OwCA, FAState2OwsCA > 
/// identifies whether these triplets have the same behavior.
///
/// Note: States can be renumerated in defeerent orders.
///

class FATestCmpDfa {

public:
    FATestCmpDfa (FAAllocatorA * pAlloc);

public:
    // all pointer except pDfa1 can be NULL
    void SetFsm1 (
            const FARSDfaCA * pDfa1,
            const FAState2OwCA * pState2Ow1,
            const FAState2OwsCA * pState2Ows1,
            const FAMealyDfaCA * pSigma1
        );
    // all pointer except pDfa2 can be NULL
    void SetFsm2 (
            const FARSDfaCA * pDfa2,
            const FAState2OwCA * pState2Ow2,
            const FAState2OwsCA * pState2Ows2,
            const FAMealyDfaCA * pSigma2
        );
    // returns true if interfaces behave identically
    const bool Process (const int MaxState, const int MaxIw);

private:
    // returns object into initial state
    void Clear ();
    // helper, returns true if state has already been procesed
    const bool WasSeen (const FABitArray * pVisited, const int State) const;
    // helper, marks state as been processed
    void SetSeen (FABitArray * pVisited, const int State);
    /// compares DFA's alphabets
    const bool CmpAlphabets ();

private:

    const FARSDfaCA * m_pDfa1;
    const FARSDfaCA * m_pDfa2;

    const FAState2OwCA * m_pState2Ow1;
    const FAState2OwCA * m_pState2Ow2;

    const FAState2OwsCA * m_pState2Ows1;
    const FAState2OwsCA * m_pState2Ows2;

    const FAMealyDfaCA * m_pSigma1;
    const FAMealyDfaCA * m_pSigma2;

    // stack of pairs: <State1, State2>, State2 \in pDfa1, State2 \in pDfa2
    FAArray_t < int > m_stack;
    // indicates whether State \in pDfa1 has already been processed
    FABitArray m_visited1;
    // indicates whether State \in pDfa2 has already been processed
    FABitArray m_visited2;
    // temporary storage for Ows1
    FAArray_cont_t < int > m_ows1;
    int * m_pOws1;
    int m_MaxOws1Count;
    // temporary storage for Ows2
    FAArray_cont_t < int > m_ows2;
    int * m_pOws2;
    int m_MaxOws2Count;
    // alphabet storage
    FAArray_cont_t < int > m_iws1;
    FAArray_cont_t < int > m_iws2;

};

}

#endif
