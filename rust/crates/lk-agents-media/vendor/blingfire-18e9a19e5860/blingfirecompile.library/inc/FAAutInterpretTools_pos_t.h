/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_INTERPRETTOOLS_POS_T_H_
#define _FA_INTERPRETTOOLS_POS_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwsCA.h"
#include "FARSNfaCA.h"
#include "FABitArray.h"
#include "FAUtils_cl.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class calculates the match between intput text positions and
/// positions in regular expression.
///

template < class Ty >
class FAAutInterpretTools_pos_t {

public:
    FAAutInterpretTools_pos_t (FAAllocatorA * pAlloc);

public:
    /// sets up AnyIw, 0 by default
    void SetAnyIw (const int AnyIw);
    /// sets RS Dfa interface
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets State -> Ows map
    void SetState2Ows (const FAState2OwsCA * pState2Ows);
    /// sets pos -> {pos} follow pos function
    void SetFollow (const FARSNfaCA * pFollow);

public:
    /// converts a chain of input weights into a chain of 
    /// regular expression positions
    /// returns false if not matched
    const bool Chain2PosChain (
            const Ty * pInChain,
            __out_ecount(Size) int * pPosChain,
            const int Size
        );

    /// the same as above but should be used for DFA with localy 
    /// expanded Any symbol
    const bool Chain2PosChain_local (
            const Ty * pInChain,
            __out_ecount(Size) int * pPosChain,
            const int Size
        );

private:
    /// returns m_AnyIw if symbol is not within the DFA's alphabet 
    /// otherwise symbol itself
    inline const int Symbol2DfaIw (const int Symbol) const;
    /// returns next matched position
    inline const int GetNextPos (
            const int * pStateSet, 
            const int StateSetSize
        ) const;
    /// stores state sequence along the path, returns the last state
    /// returns false if breaks
    inline const bool Chain2States (
            const Ty * pInChain, 
            __out_ecount(Size) int * pStates, 
            const int Size
        ) const;
    /// the same as above but for locally expanded DFAs
    inline const bool Chain2States_local (
            const Ty * pInChain,
            __out_ecount(Size) int * pStates,
            const int Size
        ) const;
    /// having DFA's states and input reconstructs match positions
    inline void States2Pos (
            const Ty * pInChain, 
            const int * pStates,
            __out_ecount(Size) int * pPosChain,
            const int Size
        );
    /// fills in m_follow array, updates m_pFollowSet and m_FollowSetMaxSize
    inline void BuildFollowSet (const int CurrPos, const int Symbol);

protected:
    const FARSDfaCA * m_pDfa;
    const FAState2OwsCA * m_pState2Ows;
    const FARSNfaCA * m_pFollow;

private:
    int m_AnyIw;
    int m_LastPos;
    FABitArray m_known_dfa_iw;
    FAArray_cont_t < int > m_ows;
    int * m_pOws;
    int m_MaxOwsSize;
    FAArray_cont_t < int > m_follow;
    int * m_pFollowSet;
    int m_FollowSetSize;
    int m_FollowSetMaxSize;
    FAArray_cont_t < int > m_Iws;
};


template < class Ty >
FAAutInterpretTools_pos_t< Ty >::
    FAAutInterpretTools_pos_t (FAAllocatorA * pAlloc) :
        m_pDfa (NULL),
        m_pState2Ows (NULL),
        m_pFollow (NULL),
        m_AnyIw (0),
        m_LastPos (-1),
        m_pOws (NULL),
        m_MaxOwsSize (0)
{
    m_known_dfa_iw.SetAllocator (pAlloc);
    m_known_dfa_iw.Create ();

    m_ows.SetAllocator (pAlloc);
    m_ows.Create ();

    m_follow.SetAllocator (pAlloc);
    m_follow.Create ();

    const int InitialFollowSetSize = 100;
    m_follow.resize (InitialFollowSetSize);

    m_FollowSetSize = 0;
    m_FollowSetMaxSize = InitialFollowSetSize;
    m_pFollowSet = m_follow.begin ();

    m_Iws.SetAllocator (pAlloc);
}


template < class Ty >
void FAAutInterpretTools_pos_t< Ty >::SetAnyIw (const int AnyIw)
{
    m_AnyIw = AnyIw;
}


template < class Ty >
void FAAutInterpretTools_pos_t< Ty >::SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pDfa = pDfa;

    if (NULL != pDfa) {

        const int Size = m_pDfa->GetIWs (NULL, 0);
        DebugLogAssert (0 < Size);

        m_Iws.Create (Size);
        m_Iws.resize (Size);
        m_pDfa->GetIWs (m_Iws.begin (), Size);

        const int * pIws = m_Iws.begin ();
        DebugLogAssert (pIws && FAIsSortUniqed (pIws, Size));

        const int MaxIw = pIws [Size - 1];

        m_known_dfa_iw.resize (MaxIw + 1);
        m_known_dfa_iw.set_bits (0, MaxIw, false);

        for (int i = 0; i < Size; ++i) {
            const int Iw = pIws [i];
            m_known_dfa_iw.set_bit (Iw, true);
        }

        m_Iws.Clear ();

    } // of if (NULL != pDfa) ...
}


template < class Ty >
void FAAutInterpretTools_pos_t< Ty >::
    SetState2Ows (const FAState2OwsCA * pState2Ows)
{
    m_pState2Ows = pState2Ows;

    if (NULL != m_pState2Ows) {

        const int MaxOwsSize = m_pState2Ows->GetMaxOwsCount ();
        m_ows.resize (MaxOwsSize);

        m_pOws = m_ows.begin ();
        m_MaxOwsSize = MaxOwsSize;
    }
}


template < class Ty >
void FAAutInterpretTools_pos_t< Ty >::SetFollow (const FARSNfaCA * pFollow)
{
    m_pFollow = pFollow;

    if (m_pFollow) {
        // get initial states
        const int * pInitials;
#ifndef NDEBUG
        const int InitialCount = 
#endif
            m_pFollow->GetInitials (&pInitials);
        // there should be only one, which is LastPos
        DebugLogAssert (1 == InitialCount && pInitials);
        m_LastPos = *pInitials;
    }
}


template < class Ty >
inline const int FAAutInterpretTools_pos_t< Ty >::
    Symbol2DfaIw (const int Symbol) const
{
    DebugLogAssert (0 <= Symbol);

    if ((unsigned int) Symbol >= m_known_dfa_iw.size ()) {

        return m_AnyIw;

    } else if (m_known_dfa_iw.get_bit (Symbol)) {

        return Symbol;

    } else {

        return m_AnyIw;
    }
}


template < class Ty >
inline const int FAAutInterpretTools_pos_t< Ty >::
    GetNextPos (
        const int * pStateSet,
        const int StateSetSize
    ) const
{
    DebugLogAssert (1 < m_FollowSetSize && m_pFollowSet);
    DebugLogAssert (FAIsSortUniqed (m_pFollowSet, m_FollowSetSize));
    DebugLogAssert (0 < StateSetSize && pStateSet);
    DebugLogAssert (FAIsSortUniqed (pStateSet, StateSetSize));

    // O (Min*log(Max))
    if (10 >= m_FollowSetSize || 10 >= StateSetSize) {

        if (StateSetSize > m_FollowSetSize) {

            int i = m_FollowSetSize - 1;
            for (; i >= 0; --i) {
                const int Pos = m_pFollowSet [i];
                if (-1 != FAFind_log (pStateSet, StateSetSize, Pos))
                    return Pos;
            }

        } else {

            int i = StateSetSize - 1;
            for (; i >= 0; --i) {
                const int Pos = pStateSet [i];
                if (-1 != FAFind_log (m_pFollowSet, m_FollowSetSize, Pos))
                    return Pos;
            }
        }

        // fatal error, intersection must exist
        DebugLogAssert (0);
        return -1;

    // O (Min+Max)
    } else {

        int i = m_FollowSetSize - 1;
        int j = StateSetSize - 1;

        while (m_pFollowSet [i] != pStateSet [j]) {

            if (m_pFollowSet [i] > pStateSet [j])
                i--;
            else
                j--;

            if (0 > i || 0 > j) {
                // fatal error, intersection must exist
                DebugLogAssert (0);
                return -1;
            }
        }

        return m_pFollowSet [i];
    }
}


template < class Ty >
inline const bool FAAutInterpretTools_pos_t< Ty >::
    Chain2States (
        const Ty * pInChain, 
        __out_ecount(Size) int * pStates, 
        const int Size
    ) const
{
    DebugLogAssert (m_pDfa && m_pState2Ows && m_pFollow);
    DebugLogAssert (0 < Size && pInChain && pStates);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        const int Symbol = pInChain [i];
        const int Iw = Symbol2DfaIw (Symbol);

        pStates [i] = State;
        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State) {
            return false;
        }
    }

    if (!m_pDfa->IsFinal (State)) {
        return false;
    }

    return true;
}


template < class Ty >
inline const bool FAAutInterpretTools_pos_t< Ty >::
    Chain2States_local (
        const Ty * pInChain, 
        __out_ecount(Size) int * pStates, 
        const int Size
    ) const
{
    DebugLogAssert (m_pDfa && m_pState2Ows && m_pFollow);
    DebugLogAssert (0 < Size && pInChain && pStates);

    int DstState;
    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < Size; ++i) {

        pStates [i] = State;

        const int Symbol = pInChain [i];
        DstState = m_pDfa->GetDest (State, Symbol);

        if (-1 == DstState) {

            State = m_pDfa->GetDest (State, m_AnyIw);

            if (-1 == State)
                return false;

        } else {

            State = DstState;
        }
    }

    if (!m_pDfa->IsFinal (State)) {
        return false;
    }

    return true;
}


template < class Ty >
inline void FAAutInterpretTools_pos_t< Ty >::
    BuildFollowSet (const int CurrPos, const int Symbol)
{
    DebugLogAssert (m_pDfa && m_pState2Ows && m_pFollow);

    int Iw = Symbol;

    // try to find transition by Symbol
    m_FollowSetSize = \
        m_pFollow->GetDest (CurrPos, Iw, m_pFollowSet, m_FollowSetMaxSize);

    // if not found, try to find transition by m_AnyIw
    if (-1 == m_FollowSetSize) {
        Iw = m_AnyIw;
        m_FollowSetSize = \
            m_pFollow->GetDest (CurrPos, Iw, m_pFollowSet, m_FollowSetMaxSize);
    } 

    // see whether buffer was not enough to copy follow set values
    if (m_FollowSetMaxSize < m_FollowSetSize) {

        m_follow.resize (m_FollowSetSize);
        m_FollowSetMaxSize = m_FollowSetSize;
        m_pFollowSet = m_follow.begin ();

        m_FollowSetSize = \
            m_pFollow->GetDest (CurrPos, Iw, m_pFollowSet, m_FollowSetMaxSize);
        DebugLogAssert (m_FollowSetMaxSize == m_FollowSetSize);
    }
}


template < class Ty >
inline void FAAutInterpretTools_pos_t< Ty >::
    States2Pos (
        const Ty * pInChain,
        const int * pStates,
        __out_ecount(Size) int * pPosChain,
        const int Size
    )
{
    DebugLogAssert (-1 != m_LastPos);
    DebugLogAssert (m_pDfa && m_pState2Ows && m_pFollow);
    DebugLogAssert (0 < Size && pInChain && pPosChain && pStates);

    int OwsCount;
    int CurrPos = m_LastPos;
    int State;

    for (int i = Size - 1; i >= 0; --i) {

        const int Symbol = pInChain [i];

        // get follow set
        BuildFollowSet (CurrPos, Symbol);
        DebugLogAssert (0 < m_FollowSetSize && m_pFollowSet);

        // see whether there is one element in
        if (1 == m_FollowSetSize) {

            CurrPos = *m_pFollowSet;

        } else {

            State = pStates [i];

            OwsCount = m_pState2Ows->GetOws (State, m_pOws, m_MaxOwsSize);
            DebugLogAssert (m_MaxOwsSize >= OwsCount);
            DebugLogAssert (0 < OwsCount);

            CurrPos = GetNextPos (m_pOws, OwsCount);
        }

        pPosChain [i] = CurrPos;

    } // of for (i = Size - 1; ...
}


template < class Ty >
const bool FAAutInterpretTools_pos_t< Ty >::
    Chain2PosChain (
        const Ty * pInChain,
        __out_ecount(Size) int * pPosChain,
        const int Size
    )
{
    DebugLogAssert (-1 != m_LastPos);
    DebugLogAssert (m_pDfa && m_pState2Ows && m_pFollow);
    DebugLogAssert (0 < Size && pInChain && pPosChain);

    int * pStates = pPosChain;

    // get states along the match path
    if (false == Chain2States (pInChain, pStates, Size))
        return false;

    // reconstruct match positions
    States2Pos (pInChain, pStates, pPosChain, Size);

    return true;
}


template < class Ty >
const bool FAAutInterpretTools_pos_t< Ty >::
    Chain2PosChain_local (
        const Ty * pInChain,
        __out_ecount(Size) int * pPosChain,
        const int Size
    )
{
    DebugLogAssert (-1 != m_LastPos);
    DebugLogAssert (m_pDfa && m_pState2Ows && m_pFollow);
    DebugLogAssert (0 < Size && pInChain && pPosChain);

    int * pStates = pPosChain;

    // get states along the match path
    if (false == Chain2States_local (pInChain, pStates, Size))
        return false;

    // reconstruct match positions
    States2Pos (pInChain, pStates, pPosChain, Size);

    return true;
}

}

#endif
