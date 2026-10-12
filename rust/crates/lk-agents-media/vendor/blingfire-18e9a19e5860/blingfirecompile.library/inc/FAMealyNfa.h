/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYNFA_H_
#define _FA_MEALYNFA_H_

#include "FAConfig.h"
#include "FAMealyNfaA.h"
#include "FAEncoder_pref.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

///
/// FSM-compile-time interafece for Mealy NFA sigma function.
///

class FAMealyNfa : public FAMealyNfaA {

public:
    FAMealyNfa (FAAllocatorA * pAlloc);
    virtual ~FAMealyNfa ();

public:
    const int GetOw (const int Src, const int Iw, const int Dst) const;
    void SetOw (const int Src, const int Iw, const int Dst, const int Ow);
    void Prepare ();
    void Clear ();

private:
    FAEncoder_pref m_enc;
    FAChain2Num_hash m_arc2ow;
};

}

#endif
