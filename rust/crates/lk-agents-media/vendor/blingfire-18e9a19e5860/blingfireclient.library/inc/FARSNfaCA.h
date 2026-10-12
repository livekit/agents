/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSNFA_CA_H_
#define _FA_RSNFA_CA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// client-side interface for Rabin-Scott NFA
///

class FARSNfaCA {

public:
    /// returns the initial states
    virtual const int GetInitials (const int ** ppStates) const = 0;
    /// returns true if the State is a final
    virtual const bool IsFinal (const int State) const = 0;
    /// for the given State and the Input weight returns Destination states 
    /// returns -1, if there is no mapping
    /// if pDstStates == NULL then method just returns the count
    virtual const int GetDest (
            const int State,
            const int Iw,
            int * pDstStates,
            const int MaxCount
        ) const = 0;
};

}

#endif
