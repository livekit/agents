/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_OW2IW_CA_H_
#define _FA_OW2IW_CA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Returns input weight and destination state for the given output weight 
/// and source state.
///
/// Note: This is basicly needed by automaton-based reversed Minimal Perfect
/// Hash (MPH), e.g. for Id -> Chain mapping.
///

class FAOw2IwCA {

public:
    /// For the given State and Ow1 returns DstState, Iw and Ow2,
    /// where Ow2 is equal or less to the specified Ow1. If no transition
    /// exist, e.g. no Ow exist less than Ow1 at the State then it returns -1.
    virtual const int GetDestIwOw (
            const int State,
            const int Ow1,
            int * pIw,
            int * pOw2
        ) const = 0;
};

}

#endif
