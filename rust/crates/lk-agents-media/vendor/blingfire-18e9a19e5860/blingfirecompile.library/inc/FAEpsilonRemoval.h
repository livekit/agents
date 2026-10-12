/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_EPSILON_REMOVAL_H_
#define _FA_EPSILON_REMOVAL_H_

#include "FAConfig.h"
#include "FAEpsilonGraph.h"
#include "FATransClosure_acyc_t.h"
#include "FARemoveUnreachable.h"
#include "FASetUtils.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;

///
/// This class removes epsilon transitions.
///
/// Note:
///  1. The most of the work is done in FAEpsilonGraph class
///  2. The algorithm makes additional treatment for AnyIw only if 
///     it has been explicitly set up.
///

class FAEpsilonRemoval {

public:
    FAEpsilonRemoval (FAAllocatorA * pAlloc);

public:
    /// sets up Nfa with epsilon transitions
    void SetInNfa (const FARSNfaA * pInNfa);
    /// sets up spacial AnyOther input weight
    void SetAnyIw (const int AnyIw);
    /// sets up the output Nfa
    void SetOutNfa (FARSNfaA * pOutNfa);
    /// specifies the Epsilon symbol
    void SetEpsilonIw (const int EpsilonIw);
    /// makes the convertion itself
    void Process ();
    /// returns object into the initial state
    void Clear ();

public:
    /// sets up custom epsilon graph, should not be used by default
    /// (it should be setup to use the same pInNFA, pOutNfa and EpsilonIw
    ///  as this object)
    void SetEpsilonGraph (FAEpsilonGraph * pG);

private:
    /// copies *m_pInNfa into *m_pOutNfa ignoring epsilon transitions
    /// makes m_has_epsilon == true, if there were epsilons found
    void CopyNoEps ();
    /// additional treatment for the AnyOther input weight
    void ProcessAnyOther ();
    /// removes epsilon transitions, from the output automaton
    void RemoveEps ();

    /// foreach iw \in Iws do 
    ///   Set[iw] = Set[iw] U DstState
    inline void BuildTransitions (const int FromState, 
				  const int * pIws, const int IwsCount, 
				  const int DstState);
    /// adds transitions into the output nfa
    inline void AddTransitions (const int State, 
				const int * pIws, 
				const int IwsCount);

private:
    bool m_has_epsilon;
    const FARSNfaA * m_pInNfa;
    FARSNfaA * m_pOutNfa;
    int m_EpsilonIw;
    /// Nfa -> graph wrapper
    FAEpsilonGraph m_eps_graph;
    /// transistive closure calculatator
    FATransClosure_acyc_t < FAEpsilonGraph > m_tr_closure;
    /// unreachable states remover
    FARemoveUnreachable m_remover;
    /// indicates whether AnyIw is set up
    bool m_process_any;
    int m_any_iw;
    /// set utils
    FASetUtils m_set_utils;
    /// custom epsilon graph
    FAEpsilonGraph * m_pG;
};

}

#endif
