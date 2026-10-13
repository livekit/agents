/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_GETIWS_CA_H_
#define _FA_GETIWS_CA_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// client-side interface for getting state's set of defined outgoing arcs
///

class FAGetIWsCA {

public:
    /// returns a set of Iws for the given Dfa state
    virtual const int GetIWs (
            const int State,
            __out_ecount_opt (MaxIwCount) int * pIws, 
            const int MaxIwCount
        ) const = 0;
    
};

}

#endif

