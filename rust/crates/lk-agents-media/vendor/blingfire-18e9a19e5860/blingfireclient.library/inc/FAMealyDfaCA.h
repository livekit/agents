/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYDFA_CA_H_
#define _FA_MEALYDFA_CA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Run-time interface for MealyDfa automaton reaction.
///

class FAMealyDfaCA {

public:
    /// For the given State and Iw returns DstState and Ow,
    /// returns -1 if transition does not exist.
    virtual const int GetDestOw (
            const int State, 
            const int Iw, 
            int * pOw
        ) const = 0;

};

}

#endif
