/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OWS_CA_H_
#define _FA_STATE2OWS_CA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// client-side interface for Multi-Moore automaton reaction
///

class FAState2OwsCA {

public:

    /// Returns number of output weights associated with the state.
    /// Returns -1, if no Ows were associated. if NULL == pOws then 
    /// does not use it, just returns the count.
    virtual const int GetOws (
            const int State,
            int * pOws,
            const int MaxCount
        ) const = 0;

    /// Returns the maximum possible size of output weights set, e.g.
    /// the maximum number of elements GetOws can return.
    virtual const int GetMaxOwsCount () const = 0;

};

}

#endif
