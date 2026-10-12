/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA_AR_JUDY_H_
#define _FA_RSDFA_AR_JUDY_H_

#include "FAConfig.h"
#include "FARSDfaA.h"
#include "FAArray_cont_t.h"
#include "FABitArray.h"
#include "FAMap_judy.h"

namespace BlingFire
{

class FAAllocatorA;


class FARSDfa_ar_judy : public FARSDfaA {

 public:

  FARSDfa_ar_judy (FAAllocatorA * pAlloc);
  virtual ~FARSDfa_ar_judy ();

 // read interface implementation
 public:

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

 // write interface implementation
 public:

  void SetMaxState (const int MaxState);
  void SetMaxIw (const int MaxIw);
  void Create ();

  void SetInitial (const int State);
  void SetFinals (const int * pStates, const int Size);
  void SetIWs (const int * pIws, const int IwsCount);
  void SetTransition (const int FromState,
                      const int Iw,
                      const int DstState);
  void SetTransition (const int FromState,
                      const int * pIws,
                      const int * pDstStates,
                      const int Count);
  void Prepare ();
  void Clear ();

 public:
  /// adds more states to the automaton,
  /// new size is OldSize + StatesCount states
  void AddStateCount (const int StatesCount);
  /// actually, just increments m_IwCount
  /// the new IwCount is old IwCount plus IwCount
  void AddIwCount (const int IwCount);

 private:
  /// creates a new state
  void create_state (const int State);

 private:

  int m_MaxIw;
  unsigned int m_StateCount;

  int m_initial;
  /// q \in m_finals iff q >= m_min_final && q <= m_max_final
  int m_min_final;
  int m_max_final;
  FAArray_cont_t < int > m_finals;

  /// alphabet of Iws
  FAArray_cont_t < int > m_alphabet;
  /// mapping: Iw -> 1 iff Iw \in m_alphabet
  FABitArray m_iw2bool;

  /// m_state_iw2dst {State -> {Iw -> Dst}},
  /// State \in 0, ..., MaxState
  FAArray_cont_t < FAMap_judy * > m_state_iw2dst;

  FAAllocatorA * m_pAlloc;
};

}

#endif
