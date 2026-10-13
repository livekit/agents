/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DICT2CLASSIFIER_H_
#define _FA_DICT2CLASSIFIER_H_

#include "FAConfig.h"
#include "FATopoSort_t.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FABitArray.h"
#include "FADfaTopoGraph.h"
#include "FASetUtils.h"

namespace BlingFire
{

class FARSDfaA;
class FAState2OwsA;
class FAAllocatorA;

///
/// Shrink suffixes in input Moore Multi Dfa (dictionary) which lead 
/// to the same reaction.
///

class FADict2Classifier {

public:
    FADict2Classifier (FAAllocatorA * pAlloc);

public:
    /// sets up input automaton
    void SetInMooreDfa (
            const FARSDfaA * pInDfa, 
            const FAState2OwsA * pInState2Ows
        );
    /// sets up output automaton
    void SetOutMooreDfa (
            FARSDfaA * pOutDfa, 
            FAState2OwsA * pOutState2Ows
        );
    /// optional, sets up Ow -> Freq array; is not used if not set
    void SetOw2Freq (const int * pOw2Freq, const int Count);
    /// sets up minimum state depth (prefix length) starting from which
    /// each state will have a reaction, DefMinDepth is used by default
    void SetMinDepth (const int MinDepth);
    /// specifies whether to extend state to output weights maps
    /// the default is true, if false then MinDepth will be ignored
    void SetExtendState2Ows (const bool ExtendState2Ows);
    /// 0 -- union (is used by default)
    /// 1 -- intersection
    /// (is valid only with conjunction with SetExtendState2Ows (true))
    void SetOwsMergeType (const int MergeType);
    /// sets up % of Ows distribution to be used for State2Ow extension
    /// for example, 50% equals median (used only if Ow2Freq is specified)
    /// (by default 100% of Ows are taken)
    void SetOwsBound (const int OwsBound);
    /// makes extended states final, false by default
    void SetExtendFinals (const bool ExtendFinals);
    /// makes convertion
    void Process ();

private:
    void Prepare ();
    void BuildDstStates (const FARSDfaA * pDfa, const int State);
    void ProcessState (const int State);
    void BuildOutput ();
    /// calculates State to (Max) Depth map (m_State2Depth)
    void CalcMaxDepth ();
    /// calculates State to (Min) Depth map (m_State2Depth)
    void CalcMinDepth ();
    /// 1. Extends reaction:
    /// foreach -1 == GetOws(q) && Depth[q] \in Limit do
    ///   SetOws(q) U= GetOws(d), s.t. d = GetDest*(q,W), W \in /.*/
    /// Where GetDest*(q,W) is GetOws(GetOws(GetOws(q, w1), ...), wn)
    /// 2. if m_ExtendFinals == true then makes extended states finals
    void ExtendFsm ();
    /// returns filtered Ows (filters according to the specified parameters)
    const int FilterOws (const int ** ppOws);


private:

    const FARSDfaA * m_pInDfa;
    const FAState2OwsA * m_pInState2Ows;
    int m_MinDepth;
    bool m_ExtendState2Ows;
    int m_MergeType;
    bool m_ExtendFinals;

    FARSDfaA * m_pOutDfa;
    FAState2OwsA * m_pOutState2Ows;

    /// if m_state2fins [State] == true then 
    ///    State leads to more than one different final states
    /// else 
    ///    State leads to a single final state
    //// ??? FABitArray m_state2fins;
    FAArray_cont_t < unsigned char > m_state2fins;
    /// if m_state2fins [State] == false then 
    ///   m_state2fin [State] == FinalState 
    /// else
    ///   m_state2fin [State] == -1
    FAArray_cont_t < int > m_state2fin;
    /// tmp storage for dst states
    FAArray_cont_t < int > m_tmp_dsts;
    /// new set of final states
    FAArray_cont_t < int > m_finals;
    /// tmp stack
    FAArray_t < int > m_stack;
    /// output states built
    FABitArray m_seen_states;
    /// in Dfa graph
    FADfaTopoGraph m_Graph;
    /// topological sorter
    FATopoSort_t < FADfaTopoGraph > m_sorter;
    /// tmp storage for different purposes
    FAArray_cont_t < int > m_tmp_arr;
    /// mapping: state -> depth (min or max)
    FAArray_cont_t < int > m_State2Depth;
    /// set utils
    FASetUtils m_sets;
    /// holds m_pOw2Freq [Ow] == Freq
    const int * m_pOw2Freq;
    int m_Ow2FreqCount;
    /// Ows distribution bound
    int m_OwsBound;
    /// every Ow with frequency greater or equal to m_MinFreqBound is accepted
    int m_MinFreqBound;

    /// allocator
    FAAllocatorA * m_pAlloc;

    enum {
        DefMinDepth = 3,
    };
};

}

#endif
