/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA_A_H_
#define _FA_RSDFA_A_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"

namespace BlingFire
{

///
/// Common interface for Rabin-Scott Dfa.
///
/// General agreement for this interface:
///   All arrays are expected to be sorted from smaller to bigger numbers,
///   if for some implementation it is not possible, this should be stated
///   explicitly in the comments.
///

class FARSDfaA : public FARSDfaCA {

/// read interface
public:

  /// returns the maximum state number, -1 if any
  virtual const int GetMaxState () const = 0;
  /// returns the maximum Iw, -1 if any
  virtual const int GetMaxIw () const = 0;
  /// returns the final states
  virtual const int GetFinals (const int ** ppStates) const = 0;
  /// returns automaton's alphabet
  virtual const int GetIWs (const int ** ppIws) const = 0;
  /// a duplicate from the FARSDfaCA
  virtual const int GetIWs (
            __out_ecount_opt (MaxIwCount) int * pIws,
            const int MaxIwCount
        ) const = 0;

/// write interface
public:

  /// sets the maximum state number, q \in [0, MaxState]
  virtual void SetMaxState (const int MaxState) = 0;
  /// sets the maximum possible state, iw \in [0, MaxIw]
  virtual void SetMaxIw (const int MaxIw) = 0;
  /// makes Dfa to be ready
  virtual void Create () = 0;
  /// adds the initial state
  virtual void SetInitial (const int State) = 0;
  /// adds final states
  virtual void SetFinals (const int * pStates, const int StateCount) = 0;
  /// sets up automaton's alphabet
  virtual void SetIWs (const int * pIws, const int IwsCount) = 0;
  /// adds a transition
  virtual void SetTransition (const int FromState,
                              const int Iw,
                              const int DstState) = 0;
  /// adds transitions
  virtual void SetTransition (const int FromState,
                              const int * pIws,
                              const int * pDstStates,
                              const int Count) = 0;
  // have to be called after all transitions have been added
  virtual void Prepare () = 0;
  /// returns container into the state as if it was just constructed
  virtual void Clear () = 0;

};

}

#endif
