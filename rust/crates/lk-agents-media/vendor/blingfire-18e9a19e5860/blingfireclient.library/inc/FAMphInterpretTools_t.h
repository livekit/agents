/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MPHINTERPRETTOOLS_T_H_
#define _FA_MPHINTERPRETTOOLS_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAMealyDfaCA.h"
#include "FAOw2IwCA.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class is a run-time interpreter for automaton-based Minimal Perfect
/// Hash (MPH). It maps Chain -> Id and Id -> Chain.
///
/// Note: Automaton's reaction should be calculated in a special way in order
/// to provide MPH, see class FARSDfa2PerfHash for details.
///

template < class Ty >
class FAMphInterpretTools_t {

public:
    FAMphInterpretTools_t ();

public:
    /// sets up RS Dfa interface
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets up MealyDfa automaton reaction interface
    void SetMealy (const FAMealyDfaCA * pMealy);
    /// sets up reverse MPH interface
    void SetOw2Iw (const FAOw2IwCA * pOw2Iw);

public:
    /// makes chain -> id, or -1 if chain is not known
    const int GetId (
            const Ty * pChain,
            const int ChainSize
        ) const;
    /// makes id -> chain, returns chain size or -1 if id is out of range
    const int GetChain (
            const int Id,
            __out_ecount_opt(MaxChainSize) Ty * pChain,
            const int MaxChainSize
        ) const;

private:
    const FARSDfaCA * m_pDfa;
    const FAMealyDfaCA * m_pMealy;
    const FAOw2IwCA * m_pOw2Iw;
};


template < class Ty >
FAMphInterpretTools_t< Ty >::
  FAMphInterpretTools_t () :
    m_pDfa (NULL),
    m_pMealy (NULL),
    m_pOw2Iw (NULL)
{
}


template < class Ty >
void FAMphInterpretTools_t< Ty >::
    SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pDfa = pDfa;
}


template < class Ty >
void FAMphInterpretTools_t< Ty >::
    SetMealy (const FAMealyDfaCA * pMealy)
{
    m_pMealy = pMealy;
}


template < class Ty >
void FAMphInterpretTools_t< Ty >::
    SetOw2Iw (const FAOw2IwCA * pOw2Iw)
{
    m_pOw2Iw = pOw2Iw;
}


template < class Ty >
const int FAMphInterpretTools_t< Ty >::
    GetId (const Ty * pChain, const int ChainSize) const
{
    DebugLogAssert (m_pDfa && m_pMealy);
    DebugLogAssert (0 < ChainSize && pChain);

    int Ow;
    int Id = 0;
    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < ChainSize; ++i) {

        const int Iw = pChain [i];
        State = m_pMealy->GetDestOw (State, Iw, &Ow);

        if (-1 == State) {
            return -1;
        }

        Id += Ow;
    }

    if (m_pDfa->IsFinal (State))
        return Id;
    else
        return -1;
}


template < class Ty >
const int FAMphInterpretTools_t< Ty >::
    GetChain (
        const int Id,
        __out_ecount_opt(MaxChainSize) Ty * pChain,
        const int MaxChainSize
    ) const
{
    DebugLogAssert (m_pDfa && m_pOw2Iw);
    DebugLogAssert (0 <= Id);

    int Iw, Ow2;
    int ChainSize = 0;
    int State = m_pDfa->GetInitial ();
    int Ow = Id;

    if(NULL != pChain) {

        while (-1 != (State = m_pOw2Iw->GetDestIwOw (State, Ow, &Iw, &Ow2))) {

            if (ChainSize < MaxChainSize) {
                pChain [ChainSize] = Iw;
            }

            ChainSize++;
            Ow -= Ow2;
        }
    } else {

        while (-1 != (State = m_pOw2Iw->GetDestIwOw (State, Ow, &Iw, &Ow2))) {

            ChainSize++;
            Ow -= Ow2;
        }
    } // of if(NULL != pChain) ...

    if (0 == Ow) {
        return ChainSize;
    } else {
        return -1;
    }
}

}

#endif
