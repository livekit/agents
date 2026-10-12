/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OW_H_
#define _FA_STATE2OW_H_

#include "FAConfig.h"
#include "FAState2OwA.h"
#include "FAMap_judy.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// separate implementation of FAState2OwA
///

class FAState2Ow : public FAState2OwA {

public:
    FAState2Ow (FAAllocatorA * pAlloc);
    virtual ~FAState2Ow ();

public:
    const int GetOw (const int State) const;
    void SetOw (const int State, const int Ow);
    void Clear ();

private:
    FAMap_judy m_state2ow;
};

}

#endif
