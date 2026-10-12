/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_RSNFA_RO_H_
#define _FA_RSNFA_RO_H_

#include "FAConfig.h"
#include "FARSNfaA.h"
#include "FAArray_cont_t.h"
#include "FANfaDelta_ro.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Read-only NFA container.
///
/// Note:
/// 1. Normally the object of this class is supposed to be loaded and used
///  and then cleared and reloaded and so on (but not modified).
/// 2. Transitions should be added in lexicographical order, see 
///  FANfaDelta_ro.h for details, FAAutIOTools guarantee that.
///

class FARSNfa_ro : public FARSNfaA {

public:
    FARSNfa_ro (FAAllocatorA * pAlloc);
    virtual ~FARSNfa_ro ();

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
    void SetInitials (const int * pStates, const int Count);
    void SetFinals (const int * pStates, const int Count);
    void Prepare ();
    void Clear ();

private:
    void SetTransition (const int FromState, const int Iw, const int DstState);

private:
    int m_MaxState;
    int m_MaxIw;
    FAArray_cont_t < int > m_initials;
    FAArray_cont_t < int > m_finals;
    FANfaDelta_ro m_delta;
};

}

#endif
