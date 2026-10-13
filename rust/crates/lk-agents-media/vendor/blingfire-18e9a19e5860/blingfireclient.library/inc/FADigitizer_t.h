/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DIGITIZER_T_H_
#define _FA_DIGITIZER_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAUtf32Utils.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Assigns a digitial value to the input chain acording to the input Moore 
/// automaton or assigns m_AnyOw.
///

template < class Ty >
class FADigitizer_t {

public:
    FADigitizer_t ();
    ~FADigitizer_t ();
    
public:
    /// sets up digitizer automaton
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets up state reaction
    void SetState2Ow (const FAState2OwCA * pState2Ow);
    /// sets up AnyOtherIw, 0 by default
    void SetAnyIw (const int AnyIw);
    /// sets up AnyOtherOw, 0 by default
    void SetAnyOw (const int AnyOw);
    /// converts input symbols into lower case
    void SetIgnoreCase (const bool IgnoreCase);

    /// this method must be called once after all automata set up
    void Prepare ();

    /// returns Ow corresponding to the Chain
    const int Process (const Ty * pChain, const int Size) const;

private:
    void Clear();   
    /// returns m_AnyIw if Iw not in m_pSymbol2Iw
    inline const int Symbol2Iw (int Symbol) const;

private:
    int m_AnyIw;
    int m_AnyOw;
    const FAState2OwCA * m_pState2Ow;
    const FARSDfaCA * m_pDfa;
    int m_MaxIw;
    int * m_pSymbol2Iw;
    bool m_IgnoreCase;
};


template < class Ty >
FADigitizer_t< Ty >::FADigitizer_t () :
    m_AnyIw (0),
    m_AnyOw (0),
    m_pState2Ow (NULL),
    m_pDfa (NULL),
    m_MaxIw (0),
    m_pSymbol2Iw (NULL),
    m_IgnoreCase (false)
{}

template < class Ty >
FADigitizer_t< Ty >::~FADigitizer_t ()
{
    FADigitizer_t< Ty >::Clear ();
}

template < class Ty >
void FADigitizer_t< Ty >::Clear ()
{
    if (m_pSymbol2Iw) {
        delete [] m_pSymbol2Iw;
        m_pSymbol2Iw = NULL;
    }
}

template < class Ty >
void FADigitizer_t< Ty >::SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pDfa = pDfa;
}


template < class Ty >
void FADigitizer_t< Ty >::SetState2Ow (const FAState2OwCA * pState2Ow)
{
    m_pState2Ow = pState2Ow;
}


template < class Ty >
void FADigitizer_t< Ty >::SetAnyIw (const int AnyIw)
{
    m_AnyIw = AnyIw;
}


template < class Ty >
void FADigitizer_t< Ty >::SetAnyOw (const int AnyOw)
{
    m_AnyOw = AnyOw;
}


template < class Ty >
void FADigitizer_t< Ty >::SetIgnoreCase (const bool IgnoreCase)
{
    m_IgnoreCase = IgnoreCase;
}


template < class Ty >
void FADigitizer_t< Ty >::Prepare ()
{
    LogAssert (m_pDfa);

    const int IwsCount = m_pDfa->GetIWs (NULL, 0);
    LogAssert (0 < IwsCount);

    int * pIws = new int [IwsCount];
    LogAssert (pIws);

    m_pDfa->GetIWs (pIws, IwsCount);
    DebugLogAssert (pIws && FAIsSortUniqed (pIws, IwsCount));

    m_MaxIw = pIws [IwsCount - 1];

    if (m_pSymbol2Iw) {
        delete [] m_pSymbol2Iw;
        m_pSymbol2Iw = NULL;
    }
    m_pSymbol2Iw = new int [m_MaxIw + 1];
    LogAssert (m_pSymbol2Iw);

    for (int Iw = 0; Iw <= m_MaxIw; ++Iw) {
        m_pSymbol2Iw [Iw] = m_AnyIw;
    }
    for (int iw_idx = 0; iw_idx < IwsCount; ++iw_idx) {
        const int Iw = pIws [iw_idx];
        m_pSymbol2Iw [Iw] = Iw;
    }

    delete [] pIws;
    pIws = NULL;
}


template < class Ty >
inline const int FADigitizer_t< Ty >::Symbol2Iw (int Symbol) const
{
    DebugLogAssert (0 <= Symbol);

    if (m_IgnoreCase) {
        Symbol = FAUtf32ToLower (Symbol);
    }

    if (0 <= Symbol && Symbol <= m_MaxIw) {

        const int Iw = m_pSymbol2Iw [Symbol];
        return Iw;

    } else {

        return m_AnyIw;
    }
}


template < class Ty >
const int FADigitizer_t< Ty >::Process (const Ty * pChain,  const int Size) const
{
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ow);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        DebugLogAssert (pChain);

        // convert input symbol into integer symbol
        const int Symbol = pChain [i];

        // convert input integer symbol into Iw
        const int Iw = Symbol2Iw (Symbol);

        // get go to the following state
        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State) {
            return m_AnyOw;
        }
    }

    // get the Ow
    const int Ow = m_pState2Ow->GetOw (State);

    if (-1 != Ow)
        return Ow;
    else
        return m_AnyOw;
}

}

#endif
