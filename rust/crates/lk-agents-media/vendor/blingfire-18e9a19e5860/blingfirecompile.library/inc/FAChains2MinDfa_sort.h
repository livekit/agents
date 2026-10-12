/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CHAINS2MINDFA_SORT_H_
#define _FA_CHAINS2MINDFA_SORT_H_

#include "FAConfig.h"
#include "FARSDfaA.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FAChain2Num_judy.h"
#include "FAHeap_t.h"
#include "FAEncoder_pref_mask.h"
#include "FABitArray.h"

namespace BlingFire
{

class FAAllocatorA;


///
/// This processor constructs Rabin-Scott Minimal Dfa from _sorted_ chains.
///
/// Note:
///
/// 1. Only const methods of FARSDfaA are implemented
/// 2. The expected sequence of usage is the following:
///
///    1. foreach Chain in Chains do this->AddChain (Chain);
///    2. this->Prepare ();
///    3. use (const FARSDfaA*) this;
///    4. this->Clear ();
///

class FAChains2MinDfa_sort : public FARSDfaA {

public:

  FAChains2MinDfa_sort (FAAllocatorA * pAlloc);
  virtual ~FAChains2MinDfa_sort ();

public:

  /// adds chains one by one
  void AddChain (const int * pChain, const int Size);

public:

  /// use the following methods for Dfa manipulation

  const int GetMaxState () const;
  const int GetMaxIw () const;
  const int GetInitial () const;
  const int GetFinals (const int ** ppStates) const;
  const int GetIWs (const int ** ppIws) const;
  const int GetIWs (
        __out_ecount_opt (MaxIwCount) int * pIws, 
        const int MaxIwCount
    ) const;
  const bool IsFinal (const int State) const;
  const int GetDest (const int State, const int Iw) const;

  // additional
  const int GetIWs (const int State, const int ** ppIws) const;

private:

  /// the following methods are not implemented

  void SetMaxState (const int MaxState);
  void SetMaxIw (const int MaxIw);
  void Create ();
  void SetInitial (const int State);
  void SetFinals (const int * pStates, const int StateCount);
  void SetIWs (const int * pIws, const int IwsCount);
  void SetTransition (const int FromState, const int Iw, const int DstState);
  void SetTransition (const int FromState, const int * pIws,
                      const int * pDstStates, const int Count);

public:

  /// makes MinDfa ready to work
  void Prepare ();
  /// returns object into the state as if it was just constructed
  void Clear ();

private:

  // adds a new state
  inline const int AddState ();
  // deletes state
  inline void DeleteState (const int State);
  // makes the state to be final
  inline void MakeFinal (const int State);

  // get the common-most prefix between the MinDfa and the Chain
  inline void CalcCommonPrefix ();
  // returns true if the State has outgoing arcs
  inline const bool HasChildren (const int State) const;
  // replaces the state by the registered one or adds it into the register
  void ReplaceOrRegister (const int State);
  // adds the remaining chain suffix
  inline void AddSuffix ();
  // returns destination state by the MaxIw
  inline const int GetMaxChild (const int State) const;
  // returns destination state or -1, during the construction
  // this method allows to add chains in any order, but not only in numerical
  inline const int GetDest_fast (const int State, const int Iw) const;
  // build state info
  inline void BuildStateInfo (const int State);
  // looks up using the state info
  inline const int * RegLookUp () const;
  // associates the State with the state info
  inline void RegAdd (const int State);
  // replaces State's transition to Child with transition to SameChild
  inline void DeleteChild (const int State,
                           const int OldChild,
                           const int NewChild);

private:

  int m_MaxIw;
  int m_MaxState;
  int m_initial;

  /// common prefix position
  int m_cp_pos;
  /// last state of the common prefix
  int m_last_state;
  /// pointer to the chain to be added
  const int * m_pChain;
  /// chain size
  int m_ChainSize;
  /// allocator
  FAAllocatorA * m_pAlloc;

  /// Maps state into the array of destination states: [Dst_1, ..., Dst_N]
  FAArray_t < FAArray_cont_t < int > * > m_state2dsts;
  /// Maps state into the following array: [0|1, Iw_1, ..., Iw_N]
  /// the first element indicates whether the state is final or not
  FAArray_t < FAArray_cont_t < int > * > m_state2iws;
  /// contiguous container for final states
  FAArray_cont_t < int > m_finals;
  /// heap of the deleted states
  FAHeap_t < int > m_deleted;
  /// register's encoder
  FAEncoder_pref_mask m_encoder;
  /// register : maps state's right language to its index
  FAChain2Num_judy m_info2state;
  /// state's language information
  FAArray_cont_t < int > m_info;

  /// stack for non-recursive version of ReplaceOrRegister
  FAArray_cont_t < int > m_stack;

  /// contiguous container for alphabet
  FAArray_cont_t < int > m_alphabet;
  /// mapping: Iw -> 1 iff Iw \in m_alphabet
  FABitArray m_iw2bool;

  /// there is 2.60 of outgoing arcs on average per state
  /// (in Natural Language Dictionary)
  enum { AVE_ARCS_PER_STATE = 4 };

};

}

#endif
