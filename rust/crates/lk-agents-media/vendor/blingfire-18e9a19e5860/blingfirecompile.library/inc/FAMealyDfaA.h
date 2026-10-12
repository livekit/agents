/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYDFA_A_H_
#define _FA_MEALYDFA_A_H_

#include "FAConfig.h"
#include "FAMealyDfaCA.h"

namespace BlingFire
{

///
/// FSM-compile-time interface for MealyDfa automaton reaction.
///

class FAMealyDfaA : public FAMealyDfaCA {

public:
    /// returns output weight for the given arc <Src, Iw>
    /// returns -1, if no Ow was associated
    virtual const int GetOw (const int Src, const int Iw) const = 0;

    /// sets up Ow for the given arc <Src, Iw>
    /// if -1 == Ow then it remove associated (if there was any) Ow
    virtual void SetOw (const int Src, const int Iw, const int Ow) = 0;

    /// have to be called after all transitions have been added
    virtual void Prepare () = 0;

    /// returns container into the state as if it was just constructed
    virtual void Clear () = 0;
};

}

#endif
