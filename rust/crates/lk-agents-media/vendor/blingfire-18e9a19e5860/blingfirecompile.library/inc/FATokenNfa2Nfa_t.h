/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TOKENNFA2NFA_H_
#define _FA_TOKENNFA2NFA_H_

#include "FAConfig.h"
#include "FAAllocatorA.h"
#include "FAMultiMapA.h"
#include "FAMultiMap_judy.h"
#include "FASetUtils.h"
#include "FAFsmConst.h"

namespace BlingFire
{

///
/// This class substitutes TokenNums (Iws) of input automaton with
///  Ows of corresponding digitizer.
///

template < class NFA >
class FATokenNfa2Nfa_t {

public:

    FATokenNfa2Nfa_t (FAAllocatorA * pAlloc);

public:
    /// sets up digitizer type, if multiple digitizers are used
    void SetDigitizer (const int Digitizer);
    /// additional iw which will be added to alphabet, same as AnyOw for digitizer
    void SetAnyIw (const int AnyIw);
    /// the base for TokenNums, other Iws should not be modified
    void SetTnBaseIw (const int TnBaseIw);
    /// the maximum value for TokenNums, other Iws should not be modified
    void SetTnMaxIw (const int TnMaxIw);
    /// input nfa with TokenNum as input weights
    void SetTokenNfa (const NFA * pTokenNfa);
    /// input map from TokenNum to CNF on TypeNum(s)
    void SetTokenNum2CNF (const FAMultiMapA * pToken2CNF);
    /// input map from TypeNum to set of output weights of the digitizer
    void SetTypeNum2Ows (const FAMultiMapA * pTypeNum2OwSet);
    /// output nfa where the Iws are Ows of the digitizer
    void SetOutNfa (NFA * pOutNfa);
    /// makes the processing itself
    void Process ();
    /// returns object into the initial state
    void Clear ();

private:
    /// 
    void Prepare ();
    /// calcs m_token_ows, it contains Ow's from digitizer only + m_AnyIw
    /// it does not contain special symbols such as LeftAnchor and RightAnchor
    void CalcOwsAlphabet ();
    /// calculates m_Token2OwSet from m_pToken2CNF and m_pTypeNum2OwSet map
    void CalcMap ();
    /// calculates Ows set by the given CNF on TypeNums
    const int CNF2OwsSet (
            const int * pCNF, 
            const int CNFSize, 
            const int ** ppOwsSet
        );
    /// sets up transitions for the given state
    void ProcessState (const int State);

private:
    /// input
    int m_Digitizer;
    int m_AnyIw;
    int m_TnBaseIw;
    int m_TnMaxIw;
    const NFA * m_pTokenNfa;
    const FAMultiMapA * m_pToken2CNF;
    const FAMultiMapA * m_pTypeNum2OwSet;
    /// output
    NFA * m_pOutNfa;
    /// maximum ow, among both token ows and non-token ows
    int m_MaxOw;
    /// Ows alphabet for Tokens
    FAArray_cont_t < int > m_token_ows;
    /// temporary n-sets container
    FAArray_cont_t < const int * > m_tmp_sets;
    FAArray_cont_t < int > m_tmp_sizes;
    /// map from TokenNum into Set of Ows
    FAMultiMap_judy m_Token2OwSet;
    /// set utility
    FASetUtils m_set_utils;
};


template < class NFA >
FATokenNfa2Nfa_t < NFA >::FATokenNfa2Nfa_t (FAAllocatorA * pAlloc) :
    m_Digitizer (FAFsmConst::DIGITIZER_TEXT),
    m_AnyIw (0),
    m_TnBaseIw (0),
    m_TnMaxIw (0),
    m_pTokenNfa (NULL),
    m_pToken2CNF (NULL),
    m_pTypeNum2OwSet (NULL),
    m_pOutNfa (NULL),
    m_MaxOw (0),
    m_set_utils (pAlloc)
{
    m_set_utils.SetResCount (4);

    m_token_ows.SetAllocator (pAlloc);
    m_token_ows.Create ();

    m_tmp_sets.SetAllocator (pAlloc);
    m_tmp_sets.Create ();

    m_tmp_sizes.SetAllocator (pAlloc);
    m_tmp_sizes.Create ();

    m_Token2OwSet.SetAllocator (pAlloc);
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::Clear ()
{
    m_token_ows.Clear ();
    m_token_ows.Create ();

    m_tmp_sets.Clear ();
    m_tmp_sets.Create ();
    
    m_tmp_sizes.Clear ();
    m_tmp_sizes.Create ();

    m_Token2OwSet.Clear ();
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::
    SetDigitizer (const int Digitizer)
{
    m_Digitizer = Digitizer;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetAnyIw (const int AnyIw)
{
    m_AnyIw = AnyIw;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetTnBaseIw (const int TnBaseIw)
{
    m_TnBaseIw = TnBaseIw;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetTnMaxIw (const int TnMaxIw)
{
    m_TnMaxIw = TnMaxIw;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetTokenNfa (const NFA * pTokenNfa)
{
    m_pTokenNfa = pTokenNfa;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetTokenNum2CNF (const FAMultiMapA * pToken2CNF)
{
    m_pToken2CNF = pToken2CNF;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetTypeNum2Ows (const FAMultiMapA * pTypeNum2OwSet)
{
    m_pTypeNum2OwSet = pTypeNum2OwSet;
}


template < class NFA >
void FATokenNfa2Nfa_t < NFA >::SetOutNfa (NFA * pOutNfa)
{
    m_pOutNfa = pOutNfa;
}


template < class NFA >
void FATokenNfa2Nfa_t< NFA >::CalcOwsAlphabet ()
{
    DebugLogAssert (m_pTypeNum2OwSet);

    m_tmp_sets.resize (0);
    m_tmp_sizes.resize (0);

    /// add all OwsSets for union

    const int * pOwsSet;
    int TypeNum = -1;
    int OwsSetSize = m_pTypeNum2OwSet->Prev (&TypeNum, &pOwsSet);

    // the map can be empty, if multiple digitizers are used
    while (-1 != OwsSetSize) {

        DebugLogAssert (0 <= TypeNum);

        if (m_Digitizer == TypeNum >> 24) {
            m_tmp_sets.push_back (pOwsSet);
            m_tmp_sizes.push_back (OwsSetSize);
        }

        /// get next OwsSet
        OwsSetSize = m_pTypeNum2OwSet->Prev (&TypeNum, &pOwsSet);
    }

    /// add Any symbol

    m_tmp_sets.push_back (&m_AnyIw);
    m_tmp_sizes.push_back (1);

    /// calc the union

    const int ** pSetArray = m_tmp_sets.begin ();
    const int * pSizeArray = m_tmp_sizes.begin ();

    m_set_utils.UnionN (pSetArray, pSizeArray, m_tmp_sizes.size (), 0);

    /// generate alphabet

    const int * pUnion;
    const int UnionSize = m_set_utils.GetRes (&pUnion, 0);

    m_token_ows.resize (UnionSize);
    memcpy (m_token_ows.begin (), pUnion, UnionSize * sizeof (int));
}


template < class NFA >
const int FATokenNfa2Nfa_t< NFA >::
    CNF2OwsSet (const int * pCNF, const int CNFSize, const int ** ppOwsSet)
{
    DebugLogAssert (ppOwsSet);

    // CNF may not exist, if multiple digitizers are used
    if (-1 == CNFSize) {

        // proecss as if it was '.'
        *ppOwsSet = m_token_ows.begin ();
        const int OwsCount = m_token_ows.size ();
        return OwsCount;
    }

    DebugLogAssert (0 < CNFSize && pCNF);

    // res0 contain the resulting OwSet
    bool Initialized = false;
    /// res1 := {}
    m_set_utils.SetRes (NULL, 0, 1);

    // make iteration thru the CNF
    for (int i = 0; i < CNFSize; ++i) {

        DebugLogAssert (pCNF);
        const int TypeNum = pCNF [i];

        if (0 < TypeNum) {

            /// get OwsSet
            const int * pOwsSet;
            const int TypeNum2 = (m_Digitizer << 24) | TypeNum;
            const int OwsSetSize = m_pTypeNum2OwSet->Get (TypeNum2, &pOwsSet);
            m_set_utils.SelfUnion (pOwsSet, OwsSetSize, 1);

        } else if (0 > TypeNum) {

            /// get OwsSet
            const int * pOwsSet;
            const int TypeNum2 = ((m_Digitizer << 24) | (-TypeNum));
            const int OwsSetSize = m_pTypeNum2OwSet->Get (TypeNum2, &pOwsSet);

            /// res2 = m_token_ows - OwsSet
            m_set_utils.Difference (m_token_ows.begin (), m_token_ows.size (), 2,
                                    pOwsSet, OwsSetSize, 3);

            const int * pDiffSet;
            const int DiffSize = m_set_utils.GetRes (&pDiffSet, 2);

            /// add the difference to the disjunction set
            m_set_utils.SelfUnion (pDiffSet, DiffSize, 1);

        } else {

            DebugLogAssert (0 == TypeNum);

            /// see whether resulting set (res0) was not initialized already
            if (false == Initialized) {

                /// res0 = res1
                const int * pOwsSet;
                const int OwsSetSize = m_set_utils.GetRes (&pOwsSet, 1);
                m_set_utils.SetRes (pOwsSet, OwsSetSize, 0);
                Initialized = true;

            } else {

                /// res0 = res0 & res1
                const int * pOwsSet;
                const int OwsSetSize = m_set_utils.GetRes (&pOwsSet, 1);
                m_set_utils.SelfIntersect (pOwsSet, OwsSetSize, 0);

            } // of if (false == Intersect)...

            /// res1 := {}
            m_set_utils.SetRes (NULL, 0, 1);

        } // of if (0 < TypeNum) ...
    } // of for (int i = 0; ...

    if (Initialized) {

        // final intersection
        const int * pRes1;
        const int Res1Size = m_set_utils.GetRes (&pRes1, 1);

        if (0 < Res1Size) {
            m_set_utils.SelfIntersect (pRes1, Res1Size, 0);
        }

        const int OwsSetSize = m_set_utils.GetRes (ppOwsSet, 0);
        return OwsSetSize;

    } else {

        // no intersection needed
        const int OwsSetSize = m_set_utils.GetRes (ppOwsSet, 1);
        return OwsSetSize;
    }
}


template < class NFA >
void FATokenNfa2Nfa_t< NFA >::CalcMap ()
{
    DebugLogAssert (m_pToken2CNF);

    m_MaxOw = 0;

    /// return the map into the initial state
    m_Token2OwSet.Clear ();

    /// assumed TokenNums are continuous
    const int MaxIw = m_pTokenNfa->GetMaxIw ();

    for (int Iw = 0; Iw <= MaxIw; ++Iw) {

        /// see whether Iw is outside the interval of TokenNums
        if (m_TnBaseIw > Iw || Iw > m_TnMaxIw) {

            /// add : Iw -> { Iw }
            m_Token2OwSet.Set (Iw, &Iw, 1);

            if (m_MaxOw < Iw)
                m_MaxOw = Iw;

        } else {

            DebugLogAssert (0 <= Iw - m_TnBaseIw);
            const int TokenNum = (m_Digitizer << 24) | (Iw - m_TnBaseIw);

            /// get next CNF
            const int * pCNF;
            const int CNFSize = m_pToken2CNF->Get (TokenNum, &pCNF);

            /// using CNF calculate OwsSet corresponding to the TokenNum
            const int * pOwsSet;
            const int OwsSetSize = CNF2OwsSet (pCNF, CNFSize, &pOwsSet);

            if (0 < OwsSetSize) {
                DebugLogAssert (pOwsSet);

                /// add : Iw -> OwsSet
                m_Token2OwSet.Set (Iw, pOwsSet, OwsSetSize);

                if (m_MaxOw < pOwsSet [OwsSetSize - 1])
                    m_MaxOw = pOwsSet [OwsSetSize - 1];
            } else {
                /// the token represents empty set
                throw FAException ("The token is empty.", __FILE__, __LINE__);
            }
        }
    } // of for
}


template < class NFA >
void FATokenNfa2Nfa_t< NFA >::Prepare ()
{
    DebugLogAssert (m_pToken2CNF);

    CalcOwsAlphabet ();

    /// calculates map from Iw to Ow, updates m_MaxOw
    CalcMap ();

    m_set_utils.SetResCount (m_MaxOw + 1);

    /// keep set utils' results containers empty
    m_set_utils.SetRes (NULL, 0, 0);
    m_set_utils.SetRes (NULL, 0, 1);
    m_set_utils.SetRes (NULL, 0, 2);
    m_set_utils.SetRes (NULL, 0, 3);
}


template < class NFA >
void FATokenNfa2Nfa_t< NFA >::ProcessState (const int State)
{
    DebugLogAssert (m_pTokenNfa);

    const int * pIWs;
    const int IwCount = m_pTokenNfa->GetIWs (State, &pIWs);

    // make iteration thru the input weights (TokenNums)
    for (int i = 0; i < IwCount; ++i) {

        DebugLogAssert (pIWs);
        const int TokenNum = pIWs [i];

        const int * pOwsSet;
        const int OwsSetSize = m_Token2OwSet.Get (TokenNum, &pOwsSet);

        // get destination set
        const int * pDstSet;
        const int DstSetSize = m_pTokenNfa->GetDest (State, TokenNum, &pDstSet);

        // add this DstSet to all ows from OwsSet
        for (int j = 0; j < OwsSetSize; ++j) {

            DebugLogAssert (pOwsSet);
            const int Ow = pOwsSet [j];

            // calc Res[ow] = Res[ow] U DstSet
            m_set_utils.SelfUnion (pDstSet, DstSetSize, Ow);
        }
    }

    // make iteration thru the _entire_ Ows alphabet, even thru the anchors
    for (int Ow = 0; Ow <= m_MaxOw; ++Ow) {

        /// get new destination set
        const int * pDstSet;
        const int DstSetSize = m_set_utils.GetRes (&pDstSet, Ow);

        /// see whether there are any transitions by the Ow
        if (0 < DstSetSize) {

            /// set up the transitions
            m_pOutNfa->SetTransition (State, Ow, pDstSet, DstSetSize);

            /// return result set into the initial state 
            m_set_utils.SetRes (NULL, 0, Ow);
        }
    }
}


template < class NFA >
void FATokenNfa2Nfa_t< NFA >::Process ()
{
    DebugLogAssert (m_pTokenNfa);
    DebugLogAssert (m_pToken2CNF);
    DebugLogAssert (m_pTypeNum2OwSet);
    DebugLogAssert (m_pOutNfa);

    /// builds weight maps and updates m_MaxOw
    Prepare ();

    /// get max ow
    m_pOutNfa->SetMaxIw (m_MaxOw);

    /// get max state
    const int MaxState = m_pTokenNfa->GetMaxState ();
    m_pOutNfa->SetMaxState (MaxState);
    m_pOutNfa->Create ();

    const int * pStates;
    int StateCount;

    /// copy initial states
    StateCount = m_pTokenNfa->GetInitials (&pStates);
    m_pOutNfa->SetInitials (pStates, StateCount);

    /// copy final states
    StateCount = m_pTokenNfa->GetFinals (&pStates);
    m_pOutNfa->SetFinals (pStates, StateCount);

    /// set up transitions
    for (int State = 0; State <= MaxState; ++State) {

        ProcessState (State);
    }

    /// make the output automaton ready
    m_pOutNfa->Prepare ();
}

}

#endif
