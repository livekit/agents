/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYNFACA_H_
#define _FA_MEALYNFACA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Client side interface for Mealy NFA reaction representation.
///

class FAMealyNfaCA {

public:
    /// returns output weight for the given arc <Src, Iw, Dst>
    /// returns -1, if no Ow was associated
    virtual const int GetOw (
            const int Src,
            const int Iw,
            const int Dst
        ) const = 0;

};

}

#endif
