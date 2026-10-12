/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA_RENUM_H_
#define _FA_RSDFA_RENUM_H_

#include "FAConfig.h"
#include "FARSDfaA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

///
/// Implements renumerated automaton having old automaton and Old2New State map
///

class FARSDfa_renum : public FARSDfaA {

public:
    FARSDfa_renum (FAAllocatorA * pAlloc);
    virtual ~FARSDfa_renum ();

public:
    void SetOldDfa (const FARSDfaA * pDfa);
    void SetOld2New (const int * pOld2NewState);

public:
    void Prepare ();
    void Clear ();

    const int GetMaxState () const;
    const int GetMaxIw () const;
    const int GetInitial () const;
    const int GetFinals (const int ** ppStates) const;
    const int GetIWs (const int ** ppIws) const;
    const int GetIWs (
            __out_ecount_opt (MaxIwCount) int * pIws, 
            const int MaxIwCount
        ) const;
    const bool IsFinal (const int State) const;
    const int GetDest (const int State, const int Iw) const;

/// not implemented
private:
    void SetMaxState (const int MaxState);
    void SetMaxIw (const int MaxIw);
    void Create ();
    void SetInitial (const int State);
    void SetFinals (const int * pStates, const int StateCount);
    void SetIWs (const int * pIws, const int IwsCount);
    void SetTransition (
            const int FromState,
            const int Iw,
            const int DstState
        );
    void SetTransition (
            const int FromState,
            const int * pIws,
            const int * pDstStates,
            const int Count
        );

private:
    const FARSDfaA * m_pDfa;
    const int * m_pOld2NewState;
    int m_NewMaxState;
    FAArray_cont_t < int > m_new2old;
    FAArray_cont_t < int > m_new_finals;
};

}

#endif
