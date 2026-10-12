/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_MEALYDFA_RO_H_
#define _FA_MEALYDFA_RO_H_

#include "FAConfig.h"
#include "FAMealyDfaA.h"
#include "FAOw2IwCA.h"
#include "FAChain2Num_hash.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSDfaCA;

///
/// Implements Read-only compile-time Mealy DFA sigma function and 
/// reversed MPH (ID --> Chain mapping).
///
/// Note: This container requires a lexicographical order of adding 
/// <Src, Iw, Ow> triplets, FAAutIOTools guaratees this order.
///

class FAMealyDfa_ro : public FAMealyDfaA, 
                      public FAOw2IwCA {

public:
    FAMealyDfa_ro (FAAllocatorA * pAlloc);
    virtual ~FAMealyDfa_ro ();

public:
    // sets up the corresponding RS DFA
    void SetRsDfa (const FARSDfaCA * pRsDfa);

// from FAOw2IwCA
public:
    const int GetDestIwOw (
            const int State,
            const int Ow1,
            int * pIw,
            int * pOw2
        ) const;

// from FAMealyDfaA
public:
    const int GetDestOw (const int State, const int Iw, int * pOw) const;
    const int GetOw (const int Src, const int Iw) const;
    void SetOw (const int Src, const int Iw, const int Ow);
    void Prepare ();
    void Clear ();

private:
    inline void AddIwsOws ();

private:
    /// State --> <IwsSetId1, OwsSetId2>
    FAArray_cont_t < int > m_State2Sets;
    /// SetId <--> Set mapping, keeps both Iws and Ows sets
    FAChain2Num_hash m_Sets;
    /// RS DFA
    const FARSDfaCA * m_pRsDfa;

    /// temporary containers

    int m_TmpState;
    FAArray_cont_t < int > m_tmp_iws;
    FAArray_cont_t < int > m_tmp_ows;
};

}

#endif
