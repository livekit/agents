/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_RSDFA_RENUM_IWS_H_
#define _FA_RSDFA_RENUM_IWS_H_

#include "FAConfig.h"
#include "FARSDfaA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAMapA;
class FAAllocatorA;

///
/// Implements renumerated automaton having more than one new input weight
/// for every old input weight. 
///
/// (This class can be used to implement the "original" automaton interface
///  by having equivalence classes and the reduced automaton.)
///

class FARSDfa_renum_iws : public FARSDfaA {

public:
    FARSDfa_renum_iws (FAAllocatorA * pAlloc);
    virtual ~FARSDfa_renum_iws ();

public:
    /// sets up an original RS-Dfa interface
    void SetOldDfa (const FARSDfaA * pDfa);
    /// sets up New Iw -> Old Iw map
    void SetNew2Old (const FAMapA * pNewIw2Old);

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
    const FAMapA * m_pNewIw2Old;
    int m_MaxIw;
    FAArray_cont_t < int > m_iws;
};

}

#endif
