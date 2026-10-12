/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_RSDFA_RO_H_
#define _FA_RSDFA_RO_H_

#include "FAConfig.h"
#include "FARSDfaA.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_hash.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Read-only DFA. The transitions should be added in the lexicographical 
/// order, FAAutIOTools guarantees that.
///

class FARSDfa_ro : public FARSDfaA {

public:
    FARSDfa_ro (FAAllocatorA * pAlloc);
    virtual ~FARSDfa_ro ();

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
    /// State --> <IwsSetId1, DstsSetId2>
    FAArray_cont_t < int > m_State2Sets;
    /// keeps SetId <--> Set mapping
    FAChain2Num_hash m_Sets;
    /// keeps alphabet
    FAArray_cont_t < int > m_iws;
    /// set of final states
    FAArray_cont_t < int > m_finals;
    /// the initial state
    int m_Initial;
    /// Max State/Iw
    int m_MaxState;
    int m_MaxIw;

    /// temporary containers

    int m_TmpState;
    FAArray_cont_t < int > m_tmp_iws;
    FAArray_cont_t < int > m_tmp_dsts;
};

}

#endif
