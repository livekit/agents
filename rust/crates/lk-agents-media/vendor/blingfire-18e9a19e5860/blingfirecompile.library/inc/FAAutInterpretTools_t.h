/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_AUTINTERPRETTOOLS_T_H_
#define _FA_AUTINTERPRETTOOLS_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAState2OwsCA.h"
#include "FASetUtils.h"
#include "FASecurity.h"

namespace BlingFire
{

template < class Ty >
class FAAutInterpretTools_t {

public:
    FAAutInterpretTools_t (FAAllocatorA * pAlloc);

public:

    /// sets up AnyIw, 0 by default
    void SetAnyIw (const int AnyIw);
    /// sets RS Dfa interface
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets State -> Ow map
    void SetState2Ow (const FAState2OwCA * pState2Ow);
    /// sets State -> Ows map
    void SetState2Ows (const FAState2OwsCA * pState2Ows);

public:

    /// returns true if RS Dfa accepts the chain
    const bool Chain2Bool (const Ty * pChain,  const int Size) const;

    /// returns ow corresponding to the input chain in the final state of RS
    const int Chain2Ow (const Ty * pChain,  const int Size) const;

    /// returns set of ows corresponding to the input chain in the final states
    /// the return value is size of set,
    /// if the return value is -1 then return pointer is undefined
    const int Chain2Ows (const Ty * pChain, 
                         const int Size, 
                         const int ** ppOws);

    /// converts chain of input weights into the chain of output weights
    void Chain2OwChain (const Ty * pChain,
                        __out_ecount(Size) int * pOwChain,
                        const int Size) const;

    /// converts chain of input weights into the chain of sets of output weights
    void Chain2OwSetChain (const Ty * pChain,
                           __out_ecount(Size) const int ** pOwsPtrs,
                           __out_ecount(Size) int * pOwsSizes,
                           const int Size);

private:

    const FARSDfaCA * m_pDfa;
    const FAState2OwCA * m_pState2Ow;
    const FAState2OwsCA * m_pState2Ows;
    int m_AnyIw;
    FASetUtils m_sets;
};


template < class Ty >
FAAutInterpretTools_t< Ty >::
    FAAutInterpretTools_t (FAAllocatorA * pAlloc) :
        m_pDfa (NULL),
        m_pState2Ow (NULL),
        m_pState2Ows (NULL),
        m_AnyIw (0),
        m_sets (pAlloc)
{
}


template < class Ty >
void FAAutInterpretTools_t< Ty >::SetAnyIw (const int AnyIw)
{
    m_AnyIw = AnyIw;
}


template < class Ty >
void FAAutInterpretTools_t< Ty >::SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pDfa = pDfa;
}


template < class Ty >
void FAAutInterpretTools_t< Ty >::SetState2Ow (const FAState2OwCA * pState2Ow)
{
    m_pState2Ow = pState2Ow;
}


template < class Ty >
void FAAutInterpretTools_t< Ty >::
    SetState2Ows (const FAState2OwsCA * pState2Ows)
{
    m_pState2Ows = pState2Ows;
}


template < class Ty >
const bool FAAutInterpretTools_t< Ty >::Chain2Bool (const Ty * pChain,  
                                                    const int Size) const
{
    DebugLogAssert (m_pDfa);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pChain);

        // convert input symbol into int
        const int Iw = pChain [i];

        // get go to the following state
        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State) {
            // no way to process Iw
            return false;
        }
    }

    if (m_pDfa->IsFinal (State))
        return true;
    else
        return false;
}


template < class Ty >
const int FAAutInterpretTools_t< Ty >::Chain2Ow (const Ty * pChain, 
                                                 const int Size) const
{
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ow);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pChain);

        // convert input symbol into int
        const int Iw = pChain [i];

        // get go to the following state
        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State)
            return -1;
    }

    // get the Ow
    const int Ow = m_pState2Ow->GetOw (State);
    return Ow;
}


template < class Ty >
const int FAAutInterpretTools_t< Ty >::Chain2Ows (const Ty * pChain,
                                                  const int Size,
                                                  const int ** ppOws)
{
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ows);
    DebugLogAssert (ppOws);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pChain);

        // convert input symbol into int
        const int Iw = pChain [i];

        // get go to the following state
        State = m_pDfa->GetDest (State, Iw);

	if (-1 == State) {
	  // no way to process Iw
	  return -1;
	}
    }

    // get the size of the reaction set
    const int OwsCount = m_pState2Ows->GetOws (State, NULL, 0);

    if (0 < OwsCount) {

        m_sets.Resize (OwsCount, 0);

        int * pSet;
        m_sets.GetRes (&pSet, 0);
        DebugLogAssert (pSet);

        // copy results
        m_pState2Ows->GetOws (State, pSet, OwsCount);

        *ppOws = pSet;
        return OwsCount;

    } else {

        *ppOws = NULL;
        return 0;
    }
}


template < class Ty >
void FAAutInterpretTools_t< Ty >::
    Chain2OwChain (
        const Ty * pChain,
        __out_ecount(Size) int * pOwChain,
        const int Size
    ) const
{
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ow);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pChain);
        DebugLogAssert (pOwChain);

        // convert input symbol into int
        const int Iw = pChain [i];

        // get go to the following state
        State = m_pDfa->GetDest (State, Iw);

	if (-1 == State) {

	  // no way for further processing
	  for (int j = i; j < Size; j++) {
	    pOwChain [j] = -1;
	  }
	  return;
	}

        const int Ow = m_pState2Ow->GetOw (State);
        pOwChain [i] = Ow;
    }
}


template < class Ty >
void FAAutInterpretTools_t< Ty >::
    Chain2OwSetChain (
        const Ty * pChain,
        __out_ecount(Size) const int ** pOwsPtrs,
        __out_ecount(Size) int * pOwsSizes,
        const int Size
    )
{
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ows);

    m_sets.SetResCount (Size);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pChain);
        DebugLogAssert (pOwsPtrs);
        DebugLogAssert (pOwsSizes);

        // convert input symbol into int
        const int Iw = pChain [i];
        // get go to the following state
        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State) {
            // no way for further processing
            for (int j = i; j < Size; j++) {
                pOwsSizes [j] = -1;
            }
            return;
        }

        // get the size of the reaction set
        const int OwsCount = m_pState2Ows->GetOws (State, NULL, 0);

        if (0 < OwsCount) {

            m_sets.Resize (OwsCount, i);

            int * pSet;
            m_sets.GetRes (&pSet, i);
            DebugLogAssert (pSet);

            // copy results
            m_pState2Ows->GetOws (State, pSet, OwsCount);

            pOwsSizes [i] = OwsCount;
            pOwsPtrs [i] = pSet;

        } else {

            pOwsSizes [i] = 0;
            pOwsPtrs [i] = NULL;
        }
    }
}

}

#endif
