/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ANY2ANYOTHER_GLOBAL_T_H_
#define _FA_ANY2ANYOTHER_GLOBAL_T_H_

#include "FAConfig.h"
#include "FASetUtils.h"
#include "FAUtils.h"
#include "FALimits.h"

namespace BlingFire
{

class FAAllocatorA;


template < class NFA_in, class NFA_out >
class FAAny2AnyOther_global_t {

public:
    FAAny2AnyOther_global_t (FAAllocatorA * pAlloc);

public:
    /// input weight to be expanded, 0 is used by default
    void SetAnyIw (const int AnyIw);
    /// makes expansion for weights >= Iw only
    /// by default expands all weights
    void SetIwBase (const int IwBase);
    /// makes expansion with weights <= Iw only
    /// by default expands all weights
    void SetIwMax (const int IwMax);
    /// sets up expansion alphabet, 
    /// 1. if -1 == ExpCount then it will be calculated (the default behavior)
    /// 2. if -1 != ExpCount then IwBase and IwMax parameters are not used
    void SetExpIws (const int * pExpIws, const int ExpCount);
    /// sets up input Nfa
    void SetInNfa (const NFA_in * pInNfa);
    /// sets up resulting Nfa
    void SetOutNfa (NFA_out * pOutNfa);
    /// makes transformation
    void Process ();

private:
    void BuildAlphabet ();
    void AddTransitions ();
    inline void CopyState (const int State);
    inline void ExpandState (const int State);

private:
    const NFA_in * m_pInNfa;
    NFA_out * m_pOutNfa;

    int m_AnyIw;
    int m_IwBase;
    int m_IwMax;

    const int * m_pExpIws;
    int m_ExpCount;

    FASetUtils m_sets;

    enum { 
        SET_ALPHABET = 0, 
        SET_DSTS, 
        SET_COUNT
    };
    enum {
        DefMinIw = 0,
        DefMaxIw = FALimits::MaxIw,
    };
};


template < class NFA_in, class NFA_out >
FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    FAAny2AnyOther_global_t (FAAllocatorA * pAlloc) :
    m_pInNfa (NULL),
    m_pOutNfa (NULL),
    m_AnyIw (0),
    m_IwBase (DefMinIw),
    m_IwMax (DefMaxIw),
    m_pExpIws (NULL),
    m_ExpCount (-1),
    m_sets (pAlloc)
{
    m_sets.SetResCount (SET_COUNT);
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    SetAnyIw (const int AnyIw)
{
    m_AnyIw = AnyIw;
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    SetIwBase (const int IwBase)
{
    m_IwBase = IwBase;
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    SetIwMax (const int IwMax)
{
    m_IwMax = IwMax;
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    SetExpIws (const int * pExpIws, const int ExpCount)
{
    m_ExpCount = ExpCount;

    if (-1 != m_ExpCount) {

        DebugLogAssert (FAIsSortUniqed (pExpIws, ExpCount));

        // makes a copy
        m_sets.SetRes (pExpIws, ExpCount, SET_ALPHABET);
        m_sets.GetRes (&m_pExpIws, SET_ALPHABET);
    }
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    SetInNfa (const NFA_in * pInNfa)
{
    m_pInNfa = pInNfa;
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    SetOutNfa (NFA_out * pOutNfa)
{
    m_pOutNfa = pOutNfa;
}

template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    BuildAlphabet ()
{
    DebugLogAssert (m_pInNfa);

    m_sets.SetRes (NULL, 0, SET_ALPHABET);

    const int MaxState = m_pInNfa->GetMaxState ();

    for (int State = 0; State <= MaxState; ++State) {

        const int * pIws;
        const int IwsCount =  m_pInNfa->GetIWs (State, &pIws);

        if (DefMaxIw == m_IwMax && DefMinIw == m_IwBase) {

            m_sets.SelfUnion (pIws, IwsCount, SET_ALPHABET);

        } else {

            for (int i = 0; i < IwsCount; ++i) {

                DebugLogAssert (pIws);
                const int Iw = pIws [i];

                if (m_IwMax >= Iw && m_IwBase <= Iw)
                    m_sets.SelfUnion (&Iw, 1, SET_ALPHABET);
            }
        }
    } // of for (int State = 0; ...
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    CopyState (const int State)
{
    DebugLogAssert (m_pInNfa && m_pOutNfa);

    const int * pIws;
    const int IwsCount =  m_pInNfa->GetIWs (State, &pIws);
    DebugLogAssert (FAIsSortUniqed (pIws, IwsCount));

    for (int i = 0; i < IwsCount; ++i) {

        DebugLogAssert (pIws);
        const int Iw = pIws [i];

        const int * pDst;
        const int DstCount = m_pInNfa->GetDest (State, Iw, &pDst);
        m_pOutNfa->SetTransition (State, Iw, pDst, DstCount);
    }
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::
    ExpandState (const int State)
{
    DebugLogAssert (m_pInNfa && m_pOutNfa);

    int i;

    const int * pAlphabet;
    const int AlphabetSize = m_sets.GetRes (&pAlphabet, SET_ALPHABET);
    DebugLogAssert (pAlphabet && 0 < AlphabetSize);

    const int * pIws;
    const int IwsCount =  m_pInNfa->GetIWs (State, &pIws);
    DebugLogAssert (FAIsSortUniqed (pIws, IwsCount));

    const int * pAnyDst;
    const int AnyDstCount = m_pInNfa->GetDest (State, m_AnyIw, &pAnyDst);
    DebugLogAssert (pAnyDst && 0 < AnyDstCount);

    for (i = 0; i < AlphabetSize; ++i) {

        const int Iw = pAlphabet [i];

        // see whether such weight did not exist
        if (-1 == FAFind_log (pIws, IwsCount, Iw)) {

            m_pOutNfa->SetTransition (State, Iw, pAnyDst, AnyDstCount);

        } else {

            const int * pDst;
            int DstCount = m_pInNfa->GetDest (State, Iw, &pDst);

            m_sets.SetRes (pDst, DstCount, SET_DSTS);
            m_sets.SelfUnion (pAnyDst, AnyDstCount, SET_DSTS);
            DstCount = m_sets.GetRes (&pDst, SET_DSTS);

            m_pOutNfa->SetTransition (State, Iw, pDst, DstCount);
        }
    } // for (int j = 0; ...

    // copy the rest
    for (i = 0; i < IwsCount; ++i) {

        DebugLogAssert (pIws);
        const int Iw = pIws [i];

        if (-1 == FAFind_log (pAlphabet, AlphabetSize, Iw)) {

            const int * pDst;
            const int DstCount = m_pInNfa->GetDest (State, Iw, &pDst);
            m_pOutNfa->SetTransition (State, Iw, pDst, DstCount);
        }
    }
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::AddTransitions ()
{
    DebugLogAssert (m_pInNfa && m_pOutNfa);

    const int MaxState = m_pInNfa->GetMaxState ();

    for (int State = 0; State <= MaxState; ++State) {

        // see whether this state has transition by m_AnyIw
        const int * pAnyDst;
        const int AnyDstCount = m_pInNfa->GetDest (State, m_AnyIw, &pAnyDst);

        if (0 < AnyDstCount)
            ExpandState (State);
        else
            CopyState (State);

    } // for (int State = 0; ... 
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_global_t< NFA_in, NFA_out >::Process ()
{
    DebugLogAssert (m_pInNfa && m_pOutNfa);

    const int * pStates;
    int Count;

    // setup MaxIw
    int MaxIw = m_pInNfa->GetMaxIw ();

    if (0 <  m_ExpCount) {

        const int Iw = m_pExpIws [m_ExpCount - 1];

        if (Iw > MaxIw) {
            MaxIw = Iw;
        }
    }

    m_pOutNfa->SetMaxIw (MaxIw);

    // setup MaxState
    const int MaxState = m_pInNfa->GetMaxState ();
    m_pOutNfa->SetMaxState (MaxState);

    // create automaton
    m_pOutNfa->Create ();

    // copy finals
    Count = m_pInNfa->GetFinals (&pStates);
    DebugLogAssert (0 < Count && pStates);
    m_pOutNfa->SetFinals (pStates, Count);

    // copy initials
    Count = m_pInNfa->GetInitials (&pStates);
    DebugLogAssert (0 < Count && pStates);
    m_pOutNfa->SetInitials (pStates, Count);

    // build expansion alphabet, if needed
    if (-1 == m_ExpCount) {
        BuildAlphabet ();
    }

    // expand, or copy transitions
    AddTransitions ();

    // make it ready to use
    m_pOutNfa->Prepare ();
}

}

#endif
