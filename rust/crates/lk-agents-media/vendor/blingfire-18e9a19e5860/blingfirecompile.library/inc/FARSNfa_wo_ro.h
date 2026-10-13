/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_RSNFA_WO_RO_H_
#define _FA_RSNFA_WO_RO_H_

#include "FAConfig.h"
#include "FARSNfaA.h"
#include "FAArray_cont_t.h"
#include "FANfaDelta_wo_ro.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Write-only/Read-only NFA container.
///

class FARSNfa_wo_ro : public FARSNfaA {

public:
    FARSNfa_wo_ro (FAAllocatorA * pAlloc);
    virtual ~FARSNfa_wo_ro ();

public:
    const int GetMaxState () const;
    const int GetMaxIw () const;
    const int GetInitials (const int ** ppStates) const;
    const int GetFinals (const int ** ppStates) const;
    const bool IsFinal (const int State) const;
    const bool IsFinal (const int * pStates, const int Count) const;
    const int GetIWs (
            const int State,
            const int ** ppIws
        ) const;
    const int GetDest (
            const int State,
            const int Iw,
            const int ** ppIwDstStates
        ) const;
    const int GetDest (
            const int State,
            const int Iw,
            int * pDstStates,
            const int MaxCount
        ) const;

public:
    void SetMaxState (const int MaxState);
    void SetMaxIw (const int MaxIw);
    void Create ();
    void SetTransition (
            const int State,
            const int Iw,
            const int * pDstStates,
            const int Count
        );
    void SetTransition (const int State, const int Iw, const int DstState);
    void SetInitials (const int * pStates, const int Count);
    void SetFinals (const int * pStates, const int Count);
    void Prepare ();
    void Clear ();

private:
    FAArray_cont_t < int > m_initials;
    FAArray_cont_t < int > m_finals;
    FANfaDelta_wo_ro m_delta;
};

}

#endif
