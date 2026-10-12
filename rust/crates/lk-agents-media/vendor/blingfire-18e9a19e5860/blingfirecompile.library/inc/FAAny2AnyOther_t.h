/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_ANY2ANYOTHER_T_H_
#define _FA_ANY2ANYOTHER_T_H_

#include "FAConfig.h"
#include "FAAllocatorA.h"
#include "FAArray_cont_t.h"
#include "FAUtils.h"

namespace BlingFire
{

/// This processor converts special symbool Any into special symbol
/// AnyOther, which makes it possible to make further determinization.
///
/// Example, input:
/// ...
/// 0 0 Any
/// 0 1 1
/// 0 1 2
/// 1 1 1
/// ...
/// output:
/// ...
/// 0 0 AnyOther
/// 0 0 1
/// 0 0 2
/// 0 1 1
/// 0 1 2
/// 1 1 1
/// ...
///

template < class NFA_in, class NFA_out >
class FAAny2AnyOther_t {

public:

  FAAny2AnyOther_t (FAAllocatorA * pAlloc);
  ~FAAny2AnyOther_t ();

public:

  // sets up the input automaton
  void SetInNfa (const NFA_in * pInNfa);
  // sets up the output automaton
  void SetOutNfa (NFA_out * pOutNfa);
  /// sets up special any symbol
  void SetAnyIw (const int AnyIw);
  /// make convertion
  void Process ();

private:

  void Prepare ();
  void ProcessState (const int State, const int * pIws, const int IwsCount);
  void ProcessAnyState (const int State, const int * pIws, const int IwsCount);
  void InitMap (const int State, const int * pIws, const int IwsCount);
  void AddTransitions (const int State, const int * pIws, const int IwsCount);

private:

  /// input automaton
  const NFA_in * m_pInNfa;
  /// output automaton
  NFA_out * m_pOutNfa;
  /// special symbol Any
  int m_AnyIw;
  /// maps input weihgts to arrays of destination sets
  FAArray_cont_t < FAArray_cont_t < int > * > m_iw2dst;
  /// allocator
  FAAllocatorA * m_pAlloc;
};


template < class NFA_in, class NFA_out >
FAAny2AnyOther_t < NFA_in, NFA_out >::
FAAny2AnyOther_t (FAAllocatorA * pAlloc) :
  m_pInNfa (NULL),
  m_pOutNfa (NULL),
  m_AnyIw (0),
  m_pAlloc (pAlloc)
{
  m_iw2dst.SetAllocator (m_pAlloc);
  m_iw2dst.Create ();
}


template < class NFA_in, class NFA_out >
FAAny2AnyOther_t < NFA_in, NFA_out >::
~FAAny2AnyOther_t ()
{
  const int Size = m_iw2dst.size ();

  for (int iw = 0; iw < Size; ++iw) {

    FAArray_cont_t < int > * pDstSet = m_iw2dst [iw];
    DebugLogAssert (pDstSet);

    pDstSet->Clear ();
    FAFree (m_pAlloc, pDstSet);
  }

  m_iw2dst.Clear ();
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::Prepare ()
{
  const int MaxIw = m_pInNfa->GetMaxIw ();
  const int OldSize = m_iw2dst.size ();

  // see whether we have to enlarge the container
  if (OldSize <= MaxIw) {

    m_iw2dst.resize (MaxIw + 1);

    for (int iw = OldSize; iw <= MaxIw; ++iw) {

      // create a new entry
      FAArray_cont_t < int > * pDstSet = (FAArray_cont_t < int > *) 
	FAAlloc (m_pAlloc, sizeof (FAArray_cont_t < int >));
      pDstSet->SetAllocator (m_pAlloc);      
      pDstSet->Create ();

      // add it
      m_iw2dst [iw] = pDstSet;
    }
  }
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::SetInNfa (const NFA_in * pInNfa)
{
  m_pInNfa = pInNfa;
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::SetOutNfa (NFA_out * pOutNfa)
{
  m_pOutNfa = pOutNfa;
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::SetAnyIw (const int AnyIw)
{
  m_AnyIw = AnyIw;
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::
InitMap (const int State, const int * pIws, const int IwsCount)
{
  DebugLogAssert (pIws);

  const int * pDstStates;

  for (int i = 0; i < IwsCount; ++i) {

    const int Iw = pIws [i];

    // get destination states
    const int DstStatesCount = m_pInNfa->GetDest (State, Iw, &pDstStates);

    if (0 < DstStatesCount) {

      DebugLogAssert (pDstStates);

      FAArray_cont_t < int > * pDstStatesSet = m_iw2dst [Iw];
      DebugLogAssert (pDstStatesSet);

      pDstStatesSet->resize (DstStatesCount);

      // copy states
      memcpy (pDstStatesSet->begin (), 
	      pDstStates, 
	      sizeof (int) * DstStatesCount);
    }
  }
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::
AddTransitions (const int State, const int * pIws, const int IwsCount)
{
  DebugLogAssert (pIws);

  for (int i = 0; i < IwsCount; ++i) {

    const int Iw = pIws [i];

    FAArray_cont_t < int > * pDstSet = m_iw2dst [Iw];
    DebugLogAssert (pDstSet);

    // sort uniq destination states
    const int NewSize = FASortUniq (pDstSet->begin (), pDstSet->end ());
    pDstSet->resize (NewSize);

    // add transitions
    m_pOutNfa->SetTransition (State, Iw, pDstSet->begin (), NewSize);

    // clear the destination set
    pDstSet->resize (0);
  }
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::
ProcessAnyState (const int State, const int * pIws, const int IwsCount)
{
  DebugLogAssert (pIws);

  // initialize { Iw -> DstSet } mapping
  InitMap (State, pIws, IwsCount);

  // modify { Iw -> DstSet } mapping
  const int * pAnyDstSet;

  const int AnyDstSetSize = 
    m_pInNfa->GetDest (State, m_AnyIw, &pAnyDstSet);

  DebugLogAssert (0 < AnyDstSetSize);
  DebugLogAssert (pAnyDstSet);

  for (int i = 0; i < AnyDstSetSize; ++i) {

    const int DstState = pAnyDstSet [i];

    for (int j = 0; j < IwsCount; ++j) {

      const int Iw = pIws [j];

      FAArray_cont_t < int > * pDstSet = m_iw2dst [Iw];
      DebugLogAssert (pDstSet);

      pDstSet->push_back (DstState);
    }
  }

  // add transitions
  AddTransitions (State, pIws, IwsCount);
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::
ProcessState (const int State, const int * pIws, const int IwsCount)
{
  const int * pDstStates;

  for (int i = 0; i < IwsCount; ++i) {

    const int Iw = pIws [i];

    // get destination states
    const int DstStatesCount = m_pInNfa->GetDest (State, Iw, &pDstStates);

    if (0 < DstStatesCount) {

      DebugLogAssert (pDstStates);

      // set up a transition
      m_pOutNfa->SetTransition (State, Iw, pDstStates, DstStatesCount);
    }
  }
}


template < class NFA_in, class NFA_out >
void FAAny2AnyOther_t < NFA_in, NFA_out >::Process ()
{
  DebugLogAssert (m_pInNfa);
  DebugLogAssert (m_pOutNfa);

  /// make processor ready to work

  Prepare ();

  /// create out automaton

  const int MaxState = m_pInNfa->GetMaxState ();
  m_pOutNfa->SetMaxState (MaxState);

  const int MaxIw = m_pInNfa->GetMaxIw ();
  m_pOutNfa->SetMaxIw (MaxIw);

  m_pOutNfa->Create ();

  /// copy initial and final states

  const int * pInitials;
  const int InitialsCount = m_pInNfa->GetInitials (&pInitials);
  m_pOutNfa->SetInitials (pInitials, InitialsCount);

  const int * pFinals;
  const int FinalsCount = m_pInNfa->GetFinals (&pFinals);
  m_pOutNfa->SetFinals (pFinals, FinalsCount);

  /// add transitions

  const int * pIws;

  // make iteration thru all the states
  for (int State = 0; State <= MaxState; ++State) {

    // get outgoing alphabet of the State
    const int IwsCount = m_pInNfa->GetIWs (State, &pIws);

    if (0 < IwsCount) {

      DebugLogAssert (pIws);

      int i = 0;

      // see whether there is Any symbol
      for (; i < IwsCount; ++i) {

	const int Iw = pIws [i];

	if (Iw == m_AnyIw)
	  break;
      }

      if (i < IwsCount) {

	// make special processing for Any
	ProcessAnyState (State, pIws, IwsCount);

      } else {

	// copy transitions as they are
	ProcessState (State, pIws, IwsCount);
      }
    }
  }

  // make out Nfa ready
  m_pOutNfa->Prepare ();
}

}

#endif
