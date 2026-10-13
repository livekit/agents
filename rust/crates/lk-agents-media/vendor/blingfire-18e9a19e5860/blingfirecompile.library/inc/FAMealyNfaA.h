/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYNFAA_H_
#define _FA_MEALYNFAA_H_

#include "FAConfig.h"
#include "FAMealyNfaCA.h"

namespace BlingFire
{

///
/// FSM-compile-time interafece for Mealy NFA sigma function.
///

class FAMealyNfaA : public FAMealyNfaCA {

public:
    /// sets up Ow for the given arc <Src, Iw, Dst>
    /// if -1 == Ow then it remove associated (if there was any) Ow
    virtual void SetOw (
            const int Src,
            const int Iw,
            const int Dst,
            const int Ow
        ) = 0;

  /// have to be called after all transitions have been added
  virtual void Prepare () = 0;

  /// returns container into the state as if it was just constructed
  virtual void Clear () = 0;

};

}

#endif
