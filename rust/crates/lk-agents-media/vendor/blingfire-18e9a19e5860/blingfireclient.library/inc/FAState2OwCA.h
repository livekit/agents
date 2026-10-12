/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OW_CA_H_
#define _FA_STATE2OW_CA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// client-side interface for Moore automaton reaction
///

class FAState2OwCA {

public:
    /// returns output weight for the given State
    /// returns -1, if no Ow was associated
    virtual const int GetOw (const int State) const = 0;

};

}

#endif
