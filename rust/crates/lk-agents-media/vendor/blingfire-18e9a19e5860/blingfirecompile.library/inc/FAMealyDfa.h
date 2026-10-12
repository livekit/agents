/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYDFA_H_
#define _FA_MEALYDFA_H_

#include "FAConfig.h"
#include "FAMealyDfaA.h"
#include "FAEncoder_pref.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FARSDfaCA;

///
/// FSM-compile-time interafece for Mealy DFA sigma function.
///

class FAMealyDfa : public FAMealyDfaA {

public:
    FAMealyDfa (FAAllocatorA * pAlloc);
    virtual ~FAMealyDfa ();

public:
    // NOTE: does not return destination state if FARSDfaCA was not setup
    const int GetDestOw (const int State, const int Iw, int * pOw) const;
    const int GetOw (const int Src, const int Iw) const;
    void SetOw (const int Src, const int Iw, const int Ow);
    void Prepare ();
    void Clear ();

public:
    void SetRsDfa (const FARSDfaCA * pRsDfa);

private:
    const FARSDfaCA * m_pRsDfa;
    FAEncoder_pref m_enc;
    FAChain2Num_hash m_arc2ow;
};

}

#endif
