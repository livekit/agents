/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RS_NFA2REVNFA_H_
#define _FA_RS_NFA2REVNFA_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FASetUtils.h"
#include "FARSNfa_wo_ro.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Reverses the automaton direction.
///

class FARSNfa2RevNfa {

public:
  FARSNfa2RevNfa (FAAllocatorA * pAlloc);

public:
  /// sets up special symbol which means any but existing at the given state
  /// (by default no special processing is added to any symbol)
  void SetAnyIw (const int AnyIw);
  /// sets up input Nfa
  void SetInNfa (const FARSNfaA * pInNfa);
  /// sets up resulting Nfa
  void SetOutNfa (FARSNfaA * pOutNfa);
  /// makes transformation
  void Process ();
  /// returns object into the initial state
  void Clear ();

private:
  // calculates the reverse Nfa without the respect to ANY transition(s)
  void RevTrivial (FARSNfaA * pOutNfa);
  // calculates additional arcs for states with ANY transition(s)
  // (some arcs need to be added, they lead to the same destination state
  //  and some arcs need to be removed, they lead to the dead-state)
  void RevAny ();

private:
  /// input pointer
  const FARSNfaA * m_pInNfa;
  /// output pointer
  FARSNfaA * m_pOutNfa;
  /// temporary NFA, is used to reverse automata with any-symbol
  FARSNfa_wo_ro m_tmp_nfa;

  /// set utility (for difference and union)
  FASetUtils m_set_utils;
  /// additional containers
  FAArray_cont_t < const int * > m_sets;
  FAArray_cont_t < int > m_set_sizes;
  FAArray_cont_t < int > m_iws;
  /// AnyIw value
  int m_any_iw;
  bool m_process_any;
};

}

#endif
