/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFA2DFA_T_H_
#define _FA_NFA2DFA_T_H_

#include "FAConfig.h"
#include "FAAllocatorA.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_judy.h"
#include "FAChain2Num_hash.h"
#include "FAEncoder_pm_dpm.h"
#include "FAState2OwsA.h"
#include "FAFsmConst.h"
#include "FAUtils.h"

#include <algorithm>

namespace BlingFire
{

///
/// This processor builds DFA from NFA.
///
/// Note: Template parameters allows to have non-const _TNfa objects, 
/// this may be usefull when _TNfa object does not exist and is just a result
/// of on-fly conversion. However, whenever possible instantiate it as 
/// FANfa2Dfa_t < const _TNfa, _TDfa >.
///

template <class _TNfa, class _TDfa>
class FANfa2Dfa_t {

public:

  FANfa2Dfa_t (FAAllocatorA * pAlloc);
  ~FANfa2Dfa_t ();

public:

  /// sets up Nfa
  void SetNFA (const _TNfa * nfa);
  /// sets up Dfa
  void SetDFA (_TDfa * dfa);
  /// sets up special symbol which means any but existing at the given state
  /// (by default no special processing is added to any symbol)
  void SetAnyIw (const int AnyIw);
  /// specifies whether it is necessary to map new states to old ones,
  /// does not do mapping by default
  void SetNew2Old (FAState2OwsA * pNew2Old);

  /// makes determinization
  void Process ();

  /// returns into the initial state, automatically called by Process
  void Clear ();

private:

  /// prepares processor to work, 
  /// this method must be called after Nfa has been initialized
  void Prepare ();

  // maps oldset into a new state
  // If such an oldset does not exist, it creates a new, 
  // otherwise returns a new state associated with the oldset.
  // Modifies m_was_added.
  const int set2state (const int * pOldset, const int Count);
  // builds map from iw to dst set
  void build_iw2dst (const int * States, const int Count);
  // marks the NewState to be final, if there are any finals withing the States
  void build_finals (const int NewState,
                     const int * pStates,
                     const int Count);
  // helper-method is called from build_iw2dst
  void sort_uniq_iw2dst ();
  // helper-method is called from sort_uniq_iw2dst
  const int sort_uniq (int * pBegin, int * pEnd);
  // helper-method is called from build_iw2dst
  void clear_iw2dst ();
  // makes additional processing for any-other symbol (if asked)
  void process_any_iw2dst(const int * pStates, const int Count);
  // adds transitions into Dfa (without any symbol)
  void add_transitions ();
  // adds transitions into Dfa (with any symbol)
  void add_transitions_any ();

private:

  /// input Nfa
  const _TNfa * m_nfa;
  /// output Dfa
  _TDfa * m_dfa;
  /// output new2old mapping
  FAState2OwsA * m_pNew2Old;

  /// state sets encoder
  FAEncoder_pm_dpm m_encoder;

  /// maps a set of old states into a new one
  FAChain2Num_judy m_oldset2state;
  // FAChain2Num_hash m_oldset2state;
  /// maps a set of old states (which contains only one state) into a new one
  FAArray_t < int > m_old2state;
  /// stack of unseen states
  FAArray_t < int * > m_stack;

  /// new final states are placed here
  FAArray_cont_t < int > m_finals;

  /// first avaliable state number
  int m_states;
  /// indicates whether a new state was created
  bool m_was_added;

  /// maps input weihgts to arrays of destination sets
  FAArray_cont_t < FAArray_cont_t < int > * > m_iw2dst;
  /// maps input weights to flags whether the arrays of destination states
  /// have to be sorted
  FAArray_cont_t < int > m_iw2need_sort;

  /// newly built state
  int m_new_state;
  /// all outgoing iws at m_new_state
  FAArray_cont_t < int > m_iws;
  /// all destination states corresponding to the m_iws
  FAArray_cont_t < int > m_dsts;

  /// if true, we should process any-other symbol
  bool m_process_any;
  /// Any value
  int m_any_iw;
  /// indicates whether m_new_state has an Any-outgoing Iw
  /// holds: If false == m_any_iw Then false == m_has_any;
  bool m_has_any;

  /// memory allocator
  FAAllocatorA * m_pAlloc;

};


template <class _TNfa, class _TDfa>
FANfa2Dfa_t<_TNfa, _TDfa>::FANfa2Dfa_t (FAAllocatorA * pAlloc) : 
    m_nfa (NULL),
    m_dfa (NULL),
    m_pNew2Old (NULL),
    m_new_state (-1),
    m_process_any (false),
    m_any_iw (-1),
    m_has_any (false),
    m_pAlloc (pAlloc)
{
    m_stack.SetAllocator (m_pAlloc);
    m_old2state.SetAllocator (m_pAlloc);
    m_iws.SetAllocator (m_pAlloc);
    m_dsts.SetAllocator (m_pAlloc);
    m_iw2dst.SetAllocator (m_pAlloc);
    m_iw2need_sort.SetAllocator (m_pAlloc);
    m_finals.SetAllocator (m_pAlloc);
    m_oldset2state.SetAllocator (m_pAlloc);
    m_oldset2state.SetEncoder (&m_encoder);
}


template <class _TNfa, class _TDfa>
FANfa2Dfa_t<_TNfa, _TDfa>::~FANfa2Dfa_t ()
{
    int i;

    const int IwCount = m_iw2dst.size ();

    for (i = 0; i < IwCount; ++i) {

        FAArray_cont_t < int > * pDstSet = m_iw2dst [i];
        DebugLogAssert (pDstSet);

        delete pDstSet;
    }

    m_iw2dst.Clear ();
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::Prepare ()
{
    DebugLogAssert (m_nfa);

    FANfa2Dfa_t::Clear ();

    const int IwCount = m_nfa->GetMaxIw () + 1;

    m_iw2need_sort.resize (IwCount);
    m_iw2dst.resize (IwCount);

    /// make initialization
    for (int i = 0; i < IwCount; ++i) {

        FAArray_cont_t < int > * pDstSet = NEW FAArray_cont_t < int >;
        DebugLogAssert (pDstSet);

        pDstSet->SetAllocator (m_pAlloc);
        /// make the default allocation
        pDstSet->Create (2);
        /// size of this array should always be >= 1
        pDstSet->resize (1);
        /// initialize the pointer
        m_iw2dst [i] = pDstSet;

        /// initialize iw 2 need_sort map
        m_iw2need_sort [i] = false;
    }
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::SetNFA (const _TNfa * nfa)
{
    DebugLogAssert (nfa);
    m_nfa = nfa;
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::SetDFA (_TDfa * dfa)
{
    DebugLogAssert (dfa);
    m_dfa = dfa;
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::SetAnyIw (const int AnyIw)
{
    m_process_any = true;
    m_any_iw = AnyIw;
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::SetNew2Old (FAState2OwsA * pNew2Old)
{
    m_pNew2Old = pNew2Old;
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::Clear ()
{
    DebugLogAssert (0 == m_stack.size ());

    m_states = 0;
    m_new_state = -1;
    m_has_any = false;

    m_stack.Clear ();
    m_stack.Create ();
    m_old2state.Clear ();
    m_old2state.Create ();
    m_finals.Clear ();
    m_finals.Create ();
    m_oldset2state.Clear ();
    m_iw2need_sort.Clear ();
    m_iw2need_sort.Create ();
    m_iws.Clear ();
    m_iws.Create ();
    m_dsts.Clear ();
    m_dsts.Create ();

    const int IwCount = m_iw2dst.size ();

    for (int i = 0; i < IwCount; ++i) {

        FAArray_cont_t < int > * pDstSet = m_iw2dst [i];
        DebugLogAssert (pDstSet);
        delete pDstSet;
    }

    m_iw2dst.Clear ();
    m_iw2dst.Create ();
}


template <class _TNfa, class _TDfa>
inline const int FANfa2Dfa_t<_TNfa, _TDfa>::set2state (const int * pOldSet,
                                                   const int Count)
{
    DebugLogAssert (NULL != pOldSet);

    // the empty set: [ 0 ] states for transition to the dead state
    DebugLogAssert (0 < Count);

    m_was_added = false;

    if (2 < Count) {

        // try to find a state
        const int * pValue = m_oldset2state.Get (pOldSet, Count);

        // it exists
        if (NULL != pValue) {

            return *pValue;
        }

        const int NewSatate = m_states;

        // add a new one
        m_oldset2state.Add (pOldSet, Count, NewSatate);

        // increment the number of states used
        m_states++;
        m_was_added = true;

        return NewSatate;

    } else if (2 == Count) {

        const int OldState = pOldSet [1];
        const int OrigSize = m_old2state.size ();

        // check whether we have to enlarge the m_old2state array
        if (OrigSize <= OldState) {

            m_old2state.resize (OldState + 1);

            // invalidate the interval from the OrigSize upto the *pOldSet
            for (int i = OrigSize; i <= OldState; ++i) {
                m_old2state [i] = -1;
            }
        }

        // try to find a state
        int NewState = m_old2state [OldState];

        // state not found
        if (-1 == NewState) {

            NewState = m_states;

            // add a new one
            m_old2state [OldState] = NewState;

            // increment the number of states used
            m_states++;
            m_was_added = true;
        }

        return NewState;

    } else {

        return FAFsmConst::DFA_DEAD_STATE;
    }
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::clear_iw2dst ()
{
    const int IwCount = m_iws.size ();

    for (int i = 0; i < IwCount; ++i) {

        const int Iw = m_iws [i];

        FAArray_cont_t < int > * pDstStates = m_iw2dst [Iw];
        DebugLogAssert (pDstStates);

        pDstStates->resize (1);

        /// for a while
        m_iw2need_sort [Iw] = false;
    }

    m_iws.resize (0);
    m_dsts.resize (0);
}


template <class _TNfa, class _TDfa>
inline const int FANfa2Dfa_t<_TNfa, _TDfa>::sort_uniq (int * pBegin, int * pEnd)
{
    DebugLogAssert (pBegin);
    DebugLogAssert (pEnd);

    std::sort (pBegin, pEnd);
    const int NewSize = int (std::unique (pBegin, pEnd) - pBegin);

    return NewSize;
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::sort_uniq_iw2dst ()
{
    const int NewSize = sort_uniq (m_iws.begin (), m_iws.end ());
    m_iws.resize (NewSize);

    const int IwCount = m_iws.size ();

    for (int i = 0; i < IwCount; ++i) {

        const int Iw = m_iws [i];
        FAArray_cont_t < int > * pDstStates = m_iw2dst [Iw];

        DebugLogAssert (pDstStates);

        const int OldSize = pDstStates->size ();

        /// check whether we have no work to do
        /// (the array has only one element or its elements are already sorted)
        if (false == m_iw2need_sort [Iw]) {

            /// make the array of [n + 1, a_1, a_2, ... a_n]
            (*pDstStates) [0] = OldSize;

        } else {

            /// arrays from one element must not be sorted
            DebugLogAssert (2 < OldSize);

            int * pBegin = pDstStates->begin () + 1;
            int * pEnd = pDstStates->end ();

            const int DstStatesNewSize = 1 + sort_uniq (pBegin, pEnd);

            pDstStates->resize (DstStatesNewSize);

            /// make the array of [n + 1, a_1, a_2, ... a_n]
            (*pDstStates) [0] = DstStatesNewSize;

            /// clear this flag
            m_iw2need_sort [Iw] = false;
        }
    }
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::
build_iw2dst (const int * pStates, const int Count)
{
    DebugLogAssert (pStates);
    DebugLogAssert (0 < Count);

    // clear the map from the prev usage
    clear_iw2dst ();

    const int * pIw;
    const int * pIwDstStates;

    // build new dst sets
    for (int i = 0; i < Count; ++i) {

        // get source state
        const int State = pStates [i];
        // get input weights
        const int IwCount = m_nfa->GetIWs (State, &pIw);

        // there is no outgoing arcs at this state
        if (0 == IwCount)
            continue;

        const int OldIwsSize = m_iws.size ();

        m_iws.resize (OldIwsSize + IwCount);
        DebugLogAssert (pIw);

        // make iteration thru its input weights
        for (int iw_idx = 0; iw_idx < IwCount; ++iw_idx) {

            // input weight
            const int Iw = pIw [iw_idx];

            // add iw
            m_iws [OldIwsSize + iw_idx] = Iw;

            // get dest set pointer
            FAArray_cont_t < int > * pDstStates = m_iw2dst [Iw];
            DebugLogAssert (pDstStates);

            const int OldSize = pDstStates->size ();

            // get the number of dest states
            const int IwDstCount = m_nfa->GetDest (State, Iw, &pIwDstStates);

            // 0 == IwDstCount iff there is an only transition to the dead state
            if (0 < IwDstCount) {

                DebugLogAssert (pIwDstStates);

                // adjust *pDstStates's size
                pDstStates->resize (OldSize + IwDstCount);

                int * pDstStatesPtr = pDstStates->begin () + OldSize;

                // append pIwDstStates into pDstStates
                memcpy (pDstStatesPtr, pIwDstStates, sizeof (int) * IwDstCount);

                // check whether there were already some States inserted
                if (1 < OldSize) {

                    // this condition can be refined in the following way:
                    // > - we have to sort and unique
                    // = - we have to unique only
                    if (*(pDstStatesPtr - 1) >= *pDstStatesPtr)
                        m_iw2need_sort [Iw] = true;
                }
            }

        } // of iteration thru the weights

    } // of iteration thru the states

    // define the order and remove doubles
    sort_uniq_iw2dst ();

    if (true == m_process_any) {

        // process any symbol if asked
        process_any_iw2dst (pStates, Count);
    }
}


///
/// Additional ANY-symbols processing to the general procedure
///
/// foreach Iw \in NewState, s.t. ANY != Iw do
///   foreach Q \in NewState do
///     if (Q, ANY, S) \in Nfa && !(Q, Iw, P) \in Nfa then
///       NewState.Dst[Iw].push S;
///
template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::
    process_any_iw2dst (const int * pStates, const int Count)
{
    DebugLogAssert (pStates);

    m_has_any = false;

    /// new state consists of only one old
    if (1 >= Count)
        return;

    const int IwsSize = m_iws.size ();
    const int * pIws = m_iws.begin ();
    DebugLogAssert (pIws);

    /// all old states have the same transition, or no transitions at all
    if (1 >= IwsSize)
        return;

    int i;

    for (i = 0; i < IwsSize; ++i) {

        const int Iw = pIws [i];

        if (Iw == m_any_iw) {

            m_has_any = true;
            break;
        }
    }

    /// no old state contains the special any symbol
    if (i == IwsSize)
        return;

    // make iteration thru the alphabet of a new state
    for (i = 0; i < IwsSize; ++i) {

        const int Iw = pIws [i];

        if (Iw == m_any_iw)
            continue;

        // get dest set pointer
        FAArray_cont_t < int > * pDstStates = m_iw2dst [Iw];
        DebugLogAssert (pDstStates);

        // make iteration thru the old states 
        for (int idx = 0; idx < Count; ++idx) {

            const int Q = pStates [idx];

            const int * pIwDst;
            const int IwDstSize = m_nfa->GetDest (Q, Iw, &pIwDst);

            // find out whether this symbol is absent
            //// 0 == IwDstSize for dead state only
            if (-1 == IwDstSize) {

                const int * pAnyDst;
                const int AnyDstSize = m_nfa->GetDest (Q, m_any_iw, &pAnyDst);

                // but there is any symbol
                if (0 < AnyDstSize) {

                    DebugLogAssert (pAnyDst);

                    // then append AnyDst to the m_iw2dst [Iw]

                    const int OldSize = pDstStates->size ();

                    // adjust *pDstStates's size
                    pDstStates->resize (OldSize + AnyDstSize);

                    int * pDstStatesPtr = pDstStates->begin () + OldSize;

                    // append pAnyDst into pDstStates
                    memcpy (pDstStatesPtr, pAnyDst, sizeof (int) * AnyDstSize);

                    if (2 < OldSize + AnyDstSize) {
                        // mark this entry to be sorted further
                        m_iw2need_sort [Iw] = true;
                    }

                } // of if (0 < AnyDstSize)

            } // of if (0 == IwDstSize)

        } // iteration thru the old states

    } // iteration thru the alphabet

    // define the order and remove doubles
    sort_uniq_iw2dst ();
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::build_finals (const int NewState,
                                          const int * pStates,
                                          const int Count)
{
    DebugLogAssert (m_nfa);

    if (true == m_nfa->IsFinal (pStates, Count)) {

        // if there is not enough place, allocates a half of existing
        m_finals.push_back (NewState);
    }
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::add_transitions ()
{
    DebugLogAssert (m_iws.size () == m_dsts.size ());

    m_dfa->SetTransition (m_new_state, m_iws.begin (),
                          m_dsts.begin (), m_iws.size ());
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::add_transitions_any ()
{
    DebugLogAssert (m_iws.size () == m_dsts.size ());

    // find Any Idx
    const int AnyIdx = FAFind_log (m_iws.begin (), m_iws.size (), m_any_iw);
    DebugLogAssert (-1 != AnyIdx);

    // get Any destination state
    const int AnyDst = m_dsts [AnyIdx];

    // put the Any transition
    m_dfa->SetTransition (m_new_state, m_any_iw, AnyDst);

    const int IwCount = m_iws.size ();

    // put all the other transitions except that comes into AnyDst
    for (int i = 0; i < IwCount; ++i) {

        const int Iw = m_iws [i];
        const int Dst = m_dsts [i];

        if (Iw != m_any_iw && Dst != AnyDst) {

            // put a transition
            m_dfa->SetTransition (m_new_state, Iw, Dst);
        }
    }
}


template <class _TNfa, class _TDfa>
void FANfa2Dfa_t<_TNfa, _TDfa>::Process ()
{
    DebugLogAssert (m_dfa && m_nfa);

    Prepare ();

    const int * pStates;
    int i;

    // build an initial state
    const int Count = m_nfa->GetInitials (&pStates);

    if (0 < Count) {

        int * pSet = (int *) FAAlloc (m_pAlloc, sizeof (int) * (Count + 2));

        pSet [1] = Count + 1;

        for (i = 2; i < int (Count + 2); ++i) {
            pSet [i] = pStates [i - 2];
        }

        // create a new state
        const int State = set2state (&(pSet [1]), Count + 1);
        DebugLogAssert (0 == State);

        // set up the initial state
        m_dfa->SetInitial (State);

        // make pSet to be [NewState, N + 1, Old_1, ..., Old_N]
        pSet [0] = State;

        // push the initial state
        m_stack.push_back (pSet);

        while (!m_stack.empty ()) {

            // get next state
            int * pFromSet = m_stack [m_stack.size () - 1];
            DebugLogAssert (pFromSet);

            // remove it from stack
            m_stack.pop_back ();

            m_new_state = pFromSet [0];
            const int FromCount = pFromSet [1] - 1;
            const int * pFromOldStates = pFromSet + 2;

            // build iw to dst set map
            build_iw2dst (pFromOldStates, FromCount);

            // build final states
            build_finals (m_new_state, pFromOldStates, FromCount);

            // build transitions
            const int IwCount = m_iws.size ();

            // make the array of destination states of the same size
            m_dsts.resize (IwCount);

            for (int iw_idx = 0; iw_idx < IwCount; ++iw_idx) {

                const int Iw = m_iws [iw_idx];
                FAArray_cont_t < int > * pDstStateSet = m_iw2dst [Iw];
                DebugLogAssert (pDstStateSet);
                DebugLogAssert (pDstStateSet->size ());
                DebugLogAssert (0 <= (*pDstStateSet) [0] && \
                  pDstStateSet->size () == (unsigned int) (*pDstStateSet) [0]);

                int * pDstSetPtr = pDstStateSet->begin ();
                const int DstSetSize = *pDstSetPtr;

                // create or find a destination state
                const int DstState = set2state (pDstSetPtr, DstSetSize);

                m_dsts [iw_idx] = DstState;

                /// check if we should add this state for further processing
                if (true == m_was_added) {

                    int * pDstState_DstSet = 
                    (int*) FAAlloc (m_pAlloc, sizeof (int) * (DstSetSize + 1));

                    // add DstState
                    pDstState_DstSet [0] = DstState;

                    // copy DstStateSet
                    memcpy (&(pDstState_DstSet [1]),
                            pDstSetPtr,
                            sizeof (int) * DstSetSize);

                    // push for processing
                    m_stack.push_back (pDstState_DstSet);
                }
            } // of for

            // add transitions into Dfa
            if (false == m_has_any) {

                add_transitions ();

            } else {

                add_transitions_any ();
            }

            if (m_pNew2Old) {

                DebugLogAssert (pFromSet);
                const int NewState = pFromSet [0];
                const int OldSetSize = pFromSet [1] - 1;
                const int * pOldSet = pFromSet + 2;

                m_pNew2Old->SetOws (NewState, pOldSet, OldSetSize);
            }

            // destroy pFromSet and forget about it
            FAFree (m_pAlloc, pFromSet);

        } // of while

    } // of if (0 < Count)

    // set up final states
    std::sort (m_finals.begin (), m_finals.end ());
    m_dfa->SetFinals (m_finals.begin (), m_finals.size ());

    // make it ready to use
    m_dfa->Prepare ();
}

}

#endif
