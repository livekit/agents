/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA_DYNAMIC_T_H_
#define _FA_RSDFA_DYNAMIC_T_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Enlarges the number of states as required.
/// Output Dfa does not have to be renumerated.
///

template < class TDfa >
class FARSDfa_dynamic_t : public TDfa {

public:
    FARSDfa_dynamic_t (FAAllocatorA * pAlloc);

/// overridden from TDfa
public:
    void SetTransition (const int FromState,
                        const int Iw,
                        const int DstState);

    void SetTransition (const int FromState,
                        const int * pIws,
                        const int * pDstStates,
                        const int Count);
};

template < class TDfa >
FARSDfa_dynamic_t< TDfa >::FARSDfa_dynamic_t (FAAllocatorA * pAlloc) :
    TDfa (pAlloc)
{}

template < class TDfa >
void FARSDfa_dynamic_t< TDfa >::SetTransition (const int FromState,
                                               const int Iw,
                                               const int DstState)
{
    int max_state = FromState;
    if (max_state < DstState)
        max_state = DstState;

    const int CurrMaxState = TDfa::GetMaxState ();

    if (max_state > CurrMaxState) {
        TDfa::AddStateCount (max_state - CurrMaxState);
    }

    const int CurrMaxIw = TDfa::GetMaxIw ();

    if (Iw > CurrMaxIw) {
        TDfa::AddIwCount (Iw - CurrMaxIw);
    }

    TDfa::SetTransition (FromState, Iw, DstState);
}

template < class TDfa >
void FARSDfa_dynamic_t< TDfa >::SetTransition (const int FromState,
                                               const int * pIws,
                                               const int * pDstStates,
                                               const int Count)
{
    int max_state = FromState;
    int max_iw = 0;

    for (int i = 0; i < Count; ++i) {

        DebugLogAssert (pDstStates);
        DebugLogAssert (pIws);

        const int DstState = pDstStates [i];
        const int Iw = pIws [i];

        if (DstState > max_state) {
            max_state = DstState;
        }
        if (Iw > max_iw) {
            max_iw = Iw;
        }
    }

    const int CurrMaxState = TDfa::GetMaxState ();

    if (max_state > CurrMaxState) {
        TDfa::AddStateCount (max_state - CurrMaxState);
    }

    const int CurrMaxIw = TDfa::GetMaxIw ();

    if (max_iw > CurrMaxIw) {
        TDfa::AddIwCount (max_iw - CurrMaxIw);
    }

    TDfa::SetTransition (FromState, pIws, pDstStates, Count);
}

}

#endif
