/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_BRRESULT_H_
#define _FA_BRRESULT_H_

#include "FAConfig.h"
#include "FABrResultA.h"
#include "FABrResultCA.h"
#include "FAMultiMap_judy.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// extracting brackets -- results container
///

class FABrResult : public FABrResultA,
                   public FABrResultCA {

public:
    FABrResult (FAAllocatorA * pAlloc);
    virtual ~FABrResult ();

public:
    /// sets up the base value which will be added to From and To
    void SetBase (const int Base);
    /// returns container into the initial state
    void Clear ();

public:
    void AddRes (const int BrId, const int From, const int To);
    const int GetRes (const int BrId, const int ** ppFromTo) const;
    const int GetNextRes (int * pBrId, const int ** ppFromTo) const;
    const int GetFrom (const int BrId) const;
    const int GetTo (const int BrId) const;

private:
    int m_Base;
    FAMultiMap_judy m_res;
};

}

#endif
