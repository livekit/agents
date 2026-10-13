/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_EXTENDCLASSIFIER_CXP_H_
#define _FA_EXTENDCLASSIFIER_CXP_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAState2Ows_ar_uniq.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaA;
class FAState2OwsA;

///
/// Extends a State -> Ows map for states without the reaction
///

class FAExtendClassifier_cxp {

public:
    FAExtendClassifier_cxp (FAAllocatorA * pAlloc);

public:
    /// sets up input guesser
    void SetFsm (const FARSDfaA * pDfa, const FAState2OwsA * pOwsMap);
    /// sets up the MaxProb corresponding to the 1.0 of the P(Class|State)
    void SetMaxProb (const int MaxProb);
    /// adds CHAIN, CLASS, COUNT statistics
    /// Note: the CHAIN is already normalized and ordered appropriatly
    void AddStat (
            const int * pChain,
            const int Size,
            const int Class,
            const int Freq
        );
    /// makes final calculations
    void Process ();
    /// returns a new State -> Ows map
    const FAState2OwsA * GetOws () const;

private:
    inline void UpdateState (const int State, const int Class, const int Freq);
    inline void Clear ();

private:
    /// input automaton
    const FARSDfaA * m_pDfa;
    /// original State -> Ows map
    const FAState2OwsA * m_pOws;
    /// max prob value
    int m_MaxProb;
    /// max input class value
    int m_MaxClass;

    /// has -1 for states which originaly have State -> Ows map entry
    /// has a cumulative Count >= 0 for states with no State -> Ows map entry
    FAArray_cont_t < int > m_State2Count;
    /// maps <State, Class> pair into a cumulative Count
    FAChain2Num_hash m_StateClass2Count;
    /// the resulting State -> Ows map
    FAState2Ows_ar_uniq m_new_ows;
    /// temporary Ows storage
    FAArray_cont_t < int > m_ows_ar;

    enum {
        DefMaxProb = 255,
    };
};

}

#endif
