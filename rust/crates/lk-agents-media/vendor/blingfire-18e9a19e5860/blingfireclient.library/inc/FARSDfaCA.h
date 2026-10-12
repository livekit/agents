/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA_CA_H_
#define _FA_RSDFA_CA_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// client-side interface for Rabin-Scott NFA
///

class FARSDfaCA {

public:
  /// returns the initial state
  virtual const int GetInitial () const = 0;
  /// returns true if the State is a final
  virtual const bool IsFinal (const int State) const = 0;
  /// for the given State and the Iw returns the Destination state
  /// returns -1 if transition does not exist
  virtual const int GetDest (const int State, const int Iw) const = 0;
  /// returns automaton's alphabet
  virtual const int GetIWs (
            __out_ecount_opt (MaxIwCount) int * pIws,
            const int MaxIwCount
        ) const = 0;
};

}

#endif
