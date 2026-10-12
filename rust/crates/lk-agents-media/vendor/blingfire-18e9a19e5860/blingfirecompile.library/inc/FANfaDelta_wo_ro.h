/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_NFADELTA_WO_RO_H_
#define _FA_NFADELTA_WO_RO_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FANfaDelta_ro.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Wright-only/Read-only Nfa delta function.
///
/// Should be used as follows:
/// 1. [ Clear ]
/// 2. AddTransition
/// 3. ...
/// 4. Prepare
/// 5. Get...
/// 6. Clear
///  and so on.
///

class FANfaDelta_wo_ro {

public:
    FANfaDelta_wo_ro (FAAllocatorA * pAlloc);

public:
    /// adds transitions
    void AddTransition (
            const int State,
            const int Iw,
            const int * pDsts,
            const int Count
        );
    /// adds a transition
    void AddTransition (
            const int State,
            const int Iw,
            const int Dst
        );
    /// makes container RO
    void Prepare ();
    /// returns into the initial state and frees memory
    void Clear ();

public:
    const int GetMaxState () const;
    const int GetMaxIw () const;
    // returns Iws for the given State
    const int GetIWs (
            const int State, 
            const int ** ppIws
        ) const;

    // returns Dsts for the given State, Iw pair
    const int GetDest (
            const int State, 
            const int Iw, 
            const int ** ppDsts
        ) const;

private:
    class _TTrCmp {
    public:
        _TTrCmp (const int * pTrs, const int Count);

    public:
        const bool operator () (const int i1, const int i2) const;

    private:
        const int * m_pTrs;
        const int m_Count;
    };

private:
    /// const delta, used in RO mode
    FANfaDelta_ro m_delta_const;
    int m_MaxIw;
    int m_MaxState;

    /// temporary, used only in WO mode
    FAArray_cont_t < int > m_tmp_tr_list;
    FAArray_cont_t < int > m_tmp_tr_idx;
    FAArray_cont_t < int > m_tmp_dst;

    // mode switch
    bool m_IsWo;
};

}

#endif
