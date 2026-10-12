/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OWS_A_H_
#define _FA_STATE2OWS_A_H_

#include "FAConfig.h"
#include "FAState2OwsCA.h"

namespace BlingFire
{

///
/// Additional interface for Moore Machines with multiple Ows at State
///

class FAState2OwsA : public FAState2OwsCA {

/// read interface
public:

    /// inherited from FAState2OwsCA
    virtual const int GetOws (
            const int State,
            int * pOws,
            const int MaxCount
        ) const = 0;

    /// returns number of out weights associated with the state
    /// returns -1, if no Ows were associated
    virtual const int GetOws (
            const int State, 
            const int ** ppOws
        ) const = 0;

/// write interface
public:

    /// re-associates output weights with the State
    virtual void SetOws (
            const int State, 
            const int * pOws, 
            const int Size
        ) = 0;
};

}

#endif
