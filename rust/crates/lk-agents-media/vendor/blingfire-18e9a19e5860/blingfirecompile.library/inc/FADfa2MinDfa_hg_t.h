/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_DFA2MINDFA_HG_T_H_
#define _FA_DFA2MINDFA_HG_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAAllocatorA.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FABitArray.h"
#include "FAChain2Num_judy.h"
#include "FAEncoder_pref_mask.h"
#include "FANfaDelta_ro.h"
#include "FAUtils.h"

#include <algorithm>

namespace BlingFire
{

//
// Hopcroft-Gries minimization algorithm, O(|Q|*log(|Q|)) complexity.
//
// Notes:
// 1. states of input Dfa are always in small letters, 
//    e.g. state, final_state, state1, state2 and so on.
// 2. n-equivalence classes of states of the input Dfa are always Capitalized
//    e.g. State, FinalState, State1, State2 and so on.
//

template < class DFA_in, class DFA_out >
class FADfa2MinDfa_hg_t {

public:
    FADfa2MinDfa_hg_t (FAAllocatorA * pAlloc);
    ~FADfa2MinDfa_hg_t ();

public:
    /// sets input dfa
    void SetInDfa (const DFA_in * pInDfa);
    /// sets output dfa
    void SetOutDfa (DFA_out * pOutDfa);
    /// makes convertion
    void Process ();
    /// mapping avaliable after procesing
    const int GetEqClass (const int state) const;
    /// returns object into the initial state, called automatically by Process
    void Clear ();

private:
    /// initializes internal structures
    void Prepare ();
    /// calculates 1-equivalence classes
    void CalcInitialEqClasses ();
    /// calculates equivalence classes
    void CalcEqClasses ();
    /// builds output dfa
    void BuildOutDfa ();

    /// methods which makes automaton to look like fully defined one

    /// returns state count of fully defined automaton
    inline const int GetMaxState_full () const;
    /// defines destination for each <state, Iw> pair
    inline const int GetDest_full (const int state, const int Iw) const;
    /// returns true if state is final
    inline const bool IsFinal_full (const int state) const;

    /// working set methods

    /// true if pair exisits
    inline const bool HasPair (const int Iw, const int ClassFrom) const;
    /// inserts a new pair into m_P
    inline void InsertPair (const int Iw, const int ClassFrom);
    /// removes pair
    inline void RemovePair (const int Iw, const int ClassFrom);

    /// sets splitting methods

    /// calculates D-set for the given partition pair
    inline void CalcDSet (const int Iw, const int ClassFrom);
    /// helper, moves state from its block/class to its block twin
    inline void MoveToTwin (const int state);
    /// splits all blocks/classes wrt states in D
    inline void SplitBlocks ();
    /// inserts new splitting pairs
    inline void UpdatePairs ();

    /// reverse delta support

    inline const int StateIw2Idx (const int state, const int Iw);
    inline const int GetIdx (const int state, const int Iw) const;
    inline void AddBackTransition (
            const int dst_state, 
            const int Iw, 
            const int src_state
        );
    inline const int GetSrcSet (
            const int state, 
            const int Iw, 
            const int ** pp_src_set
        ) const;

private:
    /// input / output dfas
    const DFA_in * m_pInDfa;
    DFA_out * m_pOutDfa;

    /// contains DFAs' alphabet
    const int * m_pAlphabet;
    int m_AlphabetSize;

    /// contains state count of the m_pInDfa
    int m_real_state_count;
    /// contains state count of fully defined m_pInDfa
    int m_state_count;

    /// partition of states into n-equivalence classes:
    /// m_elements - all the states
    /// m_B_from - maps state to the first position in m_elements of its class
    /// m_B_size - class' size m_B_size[m_B_from[state1]]
    FAArray_cont_t < int > m_B_from;
    FAArray_cont_t < int > m_B_size;
    FAArray_cont_t < int > m_elements;
    /// reversed m_elements, e.g. i == m_e2i [m_elements [i]]
    FAArray_cont_t < int > m_e2i;

    /// m_P - pairs (m_B_from[i], Iw) over which partitioning should be done
    /// m_PairCount - the number of elements in m_P, 
    ///   all pairs with idx upto m_PairCount - 1 are valid and stored in m_P
    /// m_idx2iw - reverse map, maps pair idx to Iw
    /// m_idx2from - reverse map, maps pair idx to m_B_from[i]
    FAChain2Num_judy m_P;
    FAArray_cont_t < int > m_idx2iw;
    FAArray_cont_t < int > m_idx2from;
    unsigned int m_PairCount;
    /// special encoder for pairs
    FAEncoder_pref_mask m_encoder;

    /// m_D - is a set of states from the input dfa which could be affected by
    ///   the current partitioning pair
    /// m_state2D - mapping: if m_state2D [dfa state] == 1 then dfa state \in D
    FAArray_cont_t < int > m_D;
    FABitArray m_state2D;
    /// m_C - is a set of classes with twins
    /// m_HasTwin - indicates whether C is in m_C has twin class made
    FAArray_cont_t < int > m_C;
    FABitArray m_HasTwin;

    /// keeps SrcSets, ordered by < State, Iw > pairs
    FAArray_t < FAArray_cont_t < int > * > m_SrcSets;
    /// keeps Rev(Delta) of input DFA
    FANfaDelta_ro m_rev_delta;

    /// m_states, m_iws - are used for finite states remapping
    FAArray_cont_t < int > m_states;
    FAArray_cont_t < int > m_iws;

    /// allocator
    FAAllocatorA * m_pAlloc;
};


template < class DFA_in, class DFA_out >
FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    FADfa2MinDfa_hg_t (FAAllocatorA * pAlloc) :
        m_pInDfa (NULL),
        m_pOutDfa (NULL),
        m_pAlphabet (NULL),
        m_AlphabetSize (-1),
        m_real_state_count (-1),
        m_state_count (-1),
        m_rev_delta (pAlloc),
        m_pAlloc (pAlloc)
{
    m_B_from.SetAllocator (m_pAlloc);
    m_B_from.Create ();

    m_B_size.SetAllocator (m_pAlloc);
    m_B_size.Create ();

    m_elements.SetAllocator (m_pAlloc);
    m_elements.Create ();

    m_e2i.SetAllocator (m_pAlloc);
    m_e2i.Create ();

    m_idx2iw.SetAllocator (m_pAlloc);
    m_idx2iw.Create ();

    m_idx2from.SetAllocator (m_pAlloc);
    m_idx2from.Create ();

    m_P.SetAllocator (m_pAlloc);
    m_P.SetEncoder (&m_encoder);

    m_D.SetAllocator (m_pAlloc);
    m_D.Create ();

    m_state2D.SetAllocator (m_pAlloc);
    m_state2D.Create ();

    m_C.SetAllocator (m_pAlloc);
    m_C.Create ();

    m_HasTwin.SetAllocator (m_pAlloc);
    m_HasTwin.Create ();

    m_SrcSets.SetAllocator (m_pAlloc);
    m_SrcSets.Create ();

    m_states.SetAllocator (m_pAlloc);
    m_states.Create ();

    m_iws.SetAllocator (m_pAlloc);
    m_iws.Create ();
}


template < class DFA_in, class DFA_out >
FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::~FADfa2MinDfa_hg_t ()
{
    const int SrcSetCount = m_SrcSets.size ();

    for (int i = 0; i < SrcSetCount; ++i) {
      FAArray_cont_t < int > * pSrcSet = m_SrcSets [i];
      DebugLogAssert (pSrcSet);
      pSrcSet->Clear ();
      FAFree (m_pAlloc, pSrcSet);
    }
    m_SrcSets.resize (0);
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::SetInDfa (const DFA_in * pInDfa)
{
    m_pInDfa = pInDfa;
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::SetOutDfa (DFA_out * pOutDfa)
{
    m_pOutDfa = pOutDfa;
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::Clear ()
{
    DebugLogAssert (0 == m_SrcSets.size ());

    m_PairCount = 0;
    m_P.Clear ();
    m_states.Clear ();
    m_states.Create ();
    m_iws.Clear ();
    m_iws.Create ();
    m_elements.Clear ();
    m_elements.Create ();
    m_e2i.Clear ();
    m_e2i.Create ();
    m_B_from.Clear ();
    m_B_from.Create ();
    m_B_size.Clear ();
    m_B_size.Create ();
    m_D.Clear ();
    m_D.Create ();
    m_state2D.Clear ();
    m_state2D.Create ();
    m_C.Clear ();
    m_C.Create ();
    m_HasTwin.Clear ();
    m_HasTwin.Create ();
    m_rev_delta.Clear ();
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::Process ()
{
    DebugLogAssert (NULL != m_pInDfa);
    DebugLogAssert (NULL != m_pOutDfa);

    Clear ();
    Prepare ();
    CalcInitialEqClasses ();
    CalcEqClasses ();
    BuildOutDfa ();
}


template < class DFA_in, class DFA_out >
inline const int FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    GetSrcSet (const int state, const int Iw, const int ** pp_src_set) const
{
    DebugLogAssert (pp_src_set);

    const int Count = m_rev_delta.GetDest (state, Iw, pp_src_set);

    if (0 < Count)
        return Count;
    else
        return 0;
}


template < class DFA_in, class DFA_out >
inline const int FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    StateIw2Idx (const int state, const int Iw) 
{
    int p [2];

    p [0] = state;
    p [1] = Iw;

    const int Idx = m_P.Add (p, 2, 0);

    return Idx;
}


template < class DFA_in, class DFA_out >
inline const int FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    GetIdx (const int state, const int Iw) const
{
    int p [2];

    p [0] = state;
    p [1] = Iw;

    const int Idx = m_P.GetIdx (p, 2);

    return Idx;
}


template < class DFA_in, class DFA_out >
inline void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    AddBackTransition (const int dst_state, const int Iw, const int src_state)
{
    const int SrcSetIdx = StateIw2Idx (dst_state, Iw);
    FAArray_cont_t < int > * pSrcSet;

    if ((unsigned int) SrcSetIdx == m_SrcSets.size ()) {

      // create a new SrcSet
      pSrcSet = (FAArray_cont_t < int > *)
        FAAlloc (m_pAlloc, sizeof (FAArray_cont_t < int >));

      pSrcSet->SetAllocator (m_pAlloc);
      pSrcSet->Create ();
      pSrcSet->push_back (src_state, 0);

      m_SrcSets.push_back (pSrcSet);

    } else {

      DebugLogAssert (0 <= SrcSetIdx && m_SrcSets.size () > (unsigned int) SrcSetIdx);

      pSrcSet = m_SrcSets [SrcSetIdx];

      /// see whether such an element already in the set
      const int * pBegin = pSrcSet->begin ();
      const int Size = pSrcSet->size ();

      DebugLogAssert (FAIsSortUniqed (pBegin, Size));
      const int Pos = FAFind_log (pBegin, Size, src_state);

      // check whether we have to insert a new element
      if (-1 == Pos) {

        DebugLogAssert (pBegin);

        // check whether needs no reordering
        if (pBegin [Size - 1] < src_state) {

            pSrcSet->push_back (src_state, (unsigned short) Size);

        } else {

            pSrcSet->push_back (src_state, (unsigned short) Size);
            std::sort (pSrcSet->begin (), pSrcSet->end ());
        }
      } // of if (pPos == pEnd) ...
    }
}


template < class DFA_in, class DFA_out >
inline const int FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    GetMaxState_full () const
{
    DebugLogAssert (m_pInDfa);

    return m_real_state_count + 1;
}


template < class DFA_in, class DFA_out >
inline const int FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    GetDest_full (const int state, const int Iw) const
{
    DebugLogAssert (m_pInDfa);

    if (state == m_real_state_count)
        return m_real_state_count;

    const int dst_state = m_pInDfa->GetDest (state, Iw);

    // assuming a dead-state is less than 0
    if (0 <= dst_state)
        return dst_state;
    else
        return m_real_state_count;
}


template < class DFA_in, class DFA_out >
inline const bool FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    IsFinal_full (const int state) const
{
    DebugLogAssert (m_pInDfa);

    if (state != m_real_state_count)
        return m_pInDfa->IsFinal (state);
    else
        return false;
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::Prepare ()
{
    DebugLogAssert (NULL != m_pInDfa);

    // keep real state count
    m_real_state_count = m_pInDfa->GetMaxState ();

    // keep the state count
    m_state_count = GetMaxState_full ();

    /// will be initialized by the 1-equiv classes calculation
    m_elements.resize (m_state_count);
    m_e2i.resize (m_state_count);
    m_B_from.resize (m_state_count);
    m_B_size.resize (m_state_count);
    m_state2D.resize (m_state_count);
    m_HasTwin.resize (m_state_count);

    m_state2D.set_bits (0, m_state_count - 1, false);
    m_HasTwin.set_bits (0, m_state_count - 1, false);

    int i;

#ifndef NDEBUG
    for (i = 0; i < m_state_count; ++i) {
        m_elements [i] = -1;
        m_e2i [i] = -1;
        m_B_from [i] = -1;
        m_B_size [i] = -1;
    }
#endif

    // get the alphabet
    m_AlphabetSize = m_pInDfa->GetIWs (&m_pAlphabet);
    DebugLogAssert (0 < m_AlphabetSize && m_pAlphabet);
    DebugLogAssert (FAIsSortUniqed (m_pAlphabet, m_AlphabetSize));

    // build back-transitions map
    for (int src_state = 0; src_state < m_state_count; ++src_state) {
        for (i = 0; i < m_AlphabetSize; ++i) {

            const int Iw = m_pAlphabet [i];
            const int dst_state = GetDest_full (src_state, Iw);
            DebugLogAssert (-1 != dst_state);
            // uses m_P for <state, Iw> --> Idx mapping
            AddBackTransition (dst_state, Iw, src_state);

        } // of for (i = 0
    } // of for (src_state = 0; ...

    // copy reverse transitions into more compact container
    for (int state = 0; state < m_state_count; ++state) {
        for (i = 0; i < m_AlphabetSize; ++i) {

            const int Iw = m_pAlphabet [i];
            const int Idx = GetIdx (state, Iw);

            if (-1 == Idx)
                continue;

            DebugLogAssert (0 <= Idx && m_SrcSets.size () > (unsigned int) Idx);

            FAArray_cont_t < int > * pSrcSet = m_SrcSets [Idx];
            DebugLogAssert (pSrcSet);

            const int * pBegin = pSrcSet->begin ();
            const int Size = pSrcSet->size ();
            m_rev_delta.AddTransition (state, Iw, pBegin, Size);

            pSrcSet->Clear ();
            FAFree (m_pAlloc, pSrcSet);
        }
    }

    m_SrcSets.Clear ();
    m_SrcSets.Create ();
    m_P.Clear ();
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    CalcInitialEqClasses ()
{
    DebugLogAssert (m_pAlphabet);
    DebugLogAssert (NULL != m_pInDfa);

    int Iw, IwIdx;
    int state;

    /// split states into two classes
    int i = 0;
    int j = m_state_count;
    for (state = 0; state < m_state_count; ++state) {

        if (IsFinal_full (state)) {
            j--;
            m_elements [j] = state;
            m_e2i [state] = j;
        } else {
            m_B_from [state] = 0;
            m_elements [i] = state;
            m_e2i [state] = i;
            i++;
        }
    }

    /// setup m_B_from map for class of final states
    const int FinalFirst = j;
    for (int idx = FinalFirst; idx < m_state_count; ++idx) {
        state = m_elements [idx];
        m_B_from [state] = FinalFirst;
    }

    /// store the size of the second class here
    j = m_state_count - j;
    DebugLogAssert (0 < i || 0 < j);

    /// add first set if there is one
    if (0 < i) {
        m_B_size [0] = i;
        /// add only the smallest one
        ///if (i < j) {
        ///    for (IwIdx = 0; IwIdx < m_AlphabetSize; ++IwIdx) {
        ///        Iw = m_pAlphabet [IwIdx];
        ///        InsertPair (Iw, 0);
        ///    }
        ///}
    } // of if (0 != i) ...
    /// add second set if there is one
    if (0 < j) {
        m_B_size [FinalFirst] = j;
        /// add only the smallest one
        ///if (j <= i) {
            for (IwIdx = 0; IwIdx < m_AlphabetSize; ++IwIdx) {
                Iw = m_pAlphabet [IwIdx];
                InsertPair (Iw, FinalFirst);
            }
        ///}
    } // of if (0 != j) ...
}


template < class DFA_in, class DFA_out >
inline const bool FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    HasPair (const int Iw, const int ClassFrom) const
{
    int pair [2];

    pair [0] = Iw;
    pair [1] = ClassFrom;
    const int * pIdx = m_P.Get (pair, 2);

    return NULL != pIdx;
}


template < class DFA_in, class DFA_out >
inline void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    InsertPair (const int Iw, const int ClassFrom)
{
    int pair [2];

    pair [0] = Iw;
    pair [1] = ClassFrom;
    m_P.Add (pair, 2, m_PairCount);

    m_idx2iw.push_back (Iw);
    m_idx2from.push_back (ClassFrom);

    m_PairCount++;
}


template < class DFA_in, class DFA_out >
inline void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    RemovePair (const int Iw, const int ClassFrom)
{
    int pair [2];

    pair [0] = Iw;
    pair [1] = ClassFrom;
    m_P.Remove (pair, 2);

    m_idx2iw.pop_back ();
    m_idx2from.pop_back ();

    m_PairCount--;
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    CalcDSet (const int Iw, const int ClassFrom)
{
    int i;
    int j;

    // clear from the prev step
    const int DSize = m_D.size ();
    for (i = 0; i < DSize; ++i) {
        const int state = m_D [i];
        m_state2D.set_bit (state, false);
    }
    m_D.resize (0);

    // make iteration thru the states of the ClassFrom class
    int src_state_count;
    const int * p_src_states;

    const int Size = m_B_size [ClassFrom];

    for (i = ClassFrom; i < ClassFrom + Size; ++i) {

        const int state = m_elements [i];
        src_state_count = GetSrcSet (state, Iw, &p_src_states);

        for (j = 0; j < src_state_count; ++j) {

            DebugLogAssert (p_src_states);
            const int src_state = p_src_states [j];

            /// it has not been already seen 
            if (false == m_state2D.get_bit (src_state)) {
                /// add affected state
                m_D.push_back (src_state);
                m_state2D.set_bit (src_state, true);
            }

        } // for (j = 0; ...
    } // of for (i = 0; ...
}


template < class DFA_in, class DFA_out >
inline void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    MoveToTwin (const int state)
{
    const int ClassFrom = m_B_from [state];
    const int ClassSize = m_B_size [ClassFrom];

    // if 1== ClassSize then after this method Class + Twin become the orinal 
    // Class as if there were no sptliiting.
    DebugLogAssert (0 < ClassSize);

    int TwinSize = 0;

    // has any splitting been done for this class/block?
    if (false == m_HasTwin.get_bit (ClassFrom)) {
        if (1 == ClassSize) {
            // skip elementary classes
            return;
        } else {
            // mark as been split
            m_HasTwin.set_bit (ClassFrom, true);
            m_C.push_back (ClassFrom);
        }
    } else {
        // get TwinClass size
        const int TwinFrom = ClassFrom + ClassSize;
        DebugLogAssert (0 <= TwinFrom && (unsigned int) TwinFrom < m_B_size.size ());
        TwinSize = m_B_size [TwinFrom];
        DebugLogAssert (0 < TwinSize);
    }
    // proceed splitting
    if (1 < ClassSize) {

        const int i = m_e2i [state] - ClassFrom;
        int * pClass = m_elements.begin () + ClassFrom;
        DebugLogAssert (i == FAFind_linear (pClass, ClassSize, state));

        const int last_state = pClass [ClassSize - 1];

        pClass [i] = last_state;
        m_e2i [last_state] = ClassFrom + i;
        pClass [ClassSize - 1] = state;
        m_e2i [state] = ClassFrom + ClassSize - 1;

        const int NewClassSize = ClassSize - 1;
        const int NewTwinSize = TwinSize + 1;

        m_B_size [ClassFrom] = NewClassSize;
        m_B_size [ClassFrom + NewClassSize] = NewTwinSize;

    // convert to the original class (this is done at most once per class)
    } else {
        // as we never split elementary blocks
        DebugLogAssert (0 < TwinSize && 1 == ClassSize);

        const int NewClassSize = TwinSize + 1;
        const int NewClassFrom = ClassFrom;

        m_B_size [NewClassFrom] = NewClassSize;
        const int * pNewClass = m_elements.begin () + NewClassFrom;

        for (int j = 0; j < NewClassSize; ++j) {
            const int q = pNewClass [j];
            m_B_from [q] = NewClassFrom;
        }

        // mark it back as unchanged, note: m_C will still contain ClassFrom
        m_HasTwin.set_bit (ClassFrom, false);

    } // of if (1 < ClassSize) ...
}


/// has O(|D|) complexity
template < class DFA_in, class DFA_out >
inline void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::SplitBlocks ()
{
    int i;

    // clean up from the prev. run of SplitBlocks; at most O(|D|)
    const int CSize = m_C.size ();
    for (i = 0; i < CSize; ++i) {
        const int Class = m_C [i];
        m_HasTwin.set_bit (Class, false);
    }
    m_C.resize (0);

    const int * pD = m_D.begin ();
    const int DSize = m_D.size ();

    for (i = 0; i < DSize; ++i) {

        DebugLogAssert (pD);
        const int state = pD [i];

        // amortized time over DSize runs is exactly O(1)
        MoveToTwin (state);

    } // of for (int i = 0; i < DSize; ...

    // update Twins' m_B_from; exactly O(|D|)
    const int NewCSize = m_C.size ();
    for (i = 0; i < NewCSize; ++i) {

        const int ClassFrom = m_C [i];
        if (false == m_HasTwin.get_bit (ClassFrom)) {
            continue;
        }
        const int ClassSize = m_B_size [ClassFrom];

        const int TwinClassFrom = ClassFrom + ClassSize;
        const int TwinClassSize = m_B_size [TwinClassFrom];
        const int * pTwinClass = m_elements.begin () + TwinClassFrom;
        DebugLogAssert (pTwinClass);

        for (int j = 0; j < TwinClassSize; ++j) {
            const int twin_state = pTwinClass [j];
            m_B_from [twin_state] = TwinClassFrom;
        }

    } // of for (i = 0; i < NewCSize; ...
}


template < class DFA_in, class DFA_out >
inline void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    UpdatePairs ()
{
    DebugLogAssert (m_pAlphabet);

    const int NewCSize = m_C.size ();

    for (int i = 0; i < NewCSize; ++i) {

        const int ClassFrom = m_C [i];
        if (false == m_HasTwin.get_bit (ClassFrom)) {
            continue;
        }
        const int ClassSize = m_B_size [ClassFrom];

        const int TwinClassFrom = ClassFrom + ClassSize;
        const int TwinClassSize = m_B_size [TwinClassFrom];

        for (int j = 0; j < m_AlphabetSize; ++j) {

            const int Iw = m_pAlphabet [j];

            if (true == HasPair (Iw, ClassFrom)) {

                InsertPair (Iw, TwinClassFrom);

            } else {
                if (ClassSize <= TwinClassSize) {
                    InsertPair (Iw, ClassFrom);
                } else {
                    InsertPair (Iw, TwinClassFrom);
                }
            }
        } // of for (int j = 0; j < m_AlphabetSize; ...
    } // of for (i = 0; i < NewCSize; ...
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::CalcEqClasses ()
{
    while (0 < m_PairCount) {

        DebugLogAssert (m_PairCount == m_idx2iw.size ());
        DebugLogAssert (m_PairCount == m_idx2from.size ());

        // pick one pair
        const int Iw = m_idx2iw [m_PairCount - 1];
        const int ClassFrom = m_idx2from [m_PairCount - 1];

        // determine splitting of all blocks wrt <Iw, ClassFrom>
        CalcDSet (Iw, ClassFrom);

        // remove <Iw, ClassFrom> splitting pair
        RemovePair (Iw, ClassFrom);

        // split each block wrt <Iw, ClassFrom>
        SplitBlocks ();

        // update set of splitting pairs according to modified classes
        UpdatePairs ();

    } // of while (0 < m_PairCount) ...
}


template < class DFA_in, class DFA_out >
void FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::BuildOutDfa ()
{
    int i, state, State, state_count;
    const int * p_states;

    DebugLogAssert (m_pInDfa);
    DebugLogAssert (m_pAlphabet);

    // make the output container ready
    const int MaxState = m_pInDfa->GetMaxState ();
    m_pOutDfa->SetMaxState (MaxState);
    const int MaxIw = m_pInDfa->GetMaxIw ();
    m_pOutDfa->SetMaxIw (MaxIw);
    m_pOutDfa->Create ();

    // remap initial state
    state = m_pInDfa->GetInitial ();
    State = m_elements [m_B_from [state]];
    m_pOutDfa->SetInitial (State);

    // remap final states
    state_count = m_pInDfa->GetFinals (&p_states);
    m_states.resize (state_count);

    for (i = 0; i < state_count; ++i) {

        DebugLogAssert (NULL != p_states);

        state = p_states [i];
        State = m_elements [m_B_from [state]];
        m_states [i] = State;
    }
    m_pOutDfa->SetFinals (m_states.begin (), state_count);

    // remap all the transitions
    for (state = 0; state <= MaxState; ++state) {

        m_states.resize (0);
        m_iws.resize (0);

        for (int idx = 0; idx < m_AlphabetSize; ++idx) {

            const int Iw = m_pAlphabet [idx];
            const int dst_state = m_pInDfa->GetDest (state, Iw);

            if (0 <= dst_state) {

                const int DstState = m_elements [m_B_from [dst_state]];

                m_states.push_back (DstState);
                m_iws.push_back (Iw);

            } else if (FAFsmConst::DFA_DEAD_STATE == dst_state) {

                m_states.push_back (FAFsmConst::DFA_DEAD_STATE);
                m_iws.push_back (Iw);
            }
        }

        /// set up remapped transitions for state
        const int TrCount = m_iws.size ();
        if (0 < TrCount) {
            DebugLogAssert (m_states.size () == (unsigned int) TrCount);
            State = m_elements [m_B_from [state]];
            m_pOutDfa->SetTransition (State, m_iws.begin (), m_states.begin (), TrCount);
        }
    }

    /// make the output automaton ready to work
    m_pOutDfa->Prepare ();
}


template < class DFA_in, class DFA_out >
const int FADfa2MinDfa_hg_t < DFA_in,  DFA_out >::
    GetEqClass (const int state) const
{
    const int State = m_elements [m_B_from [state]];
    return State;
}

}

#endif
