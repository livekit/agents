/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_RSDFA_WO_RO_H_
#define _FA_RSDFA_WO_RO_H_

#include "FAConfig.h"
#include "FARSDfaA.h"
#include "FAArray_cont_t.h"
#include "FARSDfa_ro.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Write-only/Read-only DFA.
///
/// Should be used as follows:
/// 1. [ Clear ]
/// 2. Set...
/// 3. ...
/// 4. Prepare
/// 5. Get...
/// 6. Clear
///  and so on.
///


class FARSDfa_wo_ro : public FARSDfaA {

public:
    FARSDfa_wo_ro (FAAllocatorA * pAlloc);
    virtual ~FARSDfa_wo_ro ();

public:
    const int GetInitial () const;
    const bool IsFinal (const int State) const;
    const int GetDest (const int State, const int Iw) const;
    const int GetIWs (const int ** ppIws) const;
    const int GetIWs (
            __out_ecount_opt (MaxIwCount) int * pIws, 
            const int MaxIwCount
        ) const;
    const int GetMaxState () const;
    const int GetMaxIw () const;
    const int GetFinals (const int ** ppStates) const;

public:
    void SetMaxState (const int MaxState);
    void SetMaxIw (const int MaxIw);
    void Create ();
    void SetInitial (const int State);
    void SetFinals (const int * pStates, const int Count);
    void SetIWs (const int * pIws, const int Count);
    void SetTransition (
            const int State, 
            const int Iw, 
            const int Dst
        );
    void SetTransition (
            const int State,
            const int * pIws,
            const int * pDsts,
            const int Count
        );
    void Prepare ();
    void Clear ();

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
    /// RO DFA
    FARSDfa_ro m_ro_dfa;
    /// temporary, used only in WO mode
    FAArray_cont_t < int > m_tmp_tr_list;
    FAArray_cont_t < int > m_tmp_tr_idx;
    // mode switch
    bool m_IsWo;
};

}

#endif
