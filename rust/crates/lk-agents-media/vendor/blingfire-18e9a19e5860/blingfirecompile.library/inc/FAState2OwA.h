/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OW_A_H_
#define _FA_STATE2OW_A_H_

#include "FAConfig.h"
#include "FAState2OwCA.h"

namespace BlingFire
{

///
/// Additional interface for Moore DFAs
///

class FAState2OwA : public FAState2OwCA {

/// write interface
public:
    /// sets up Ow to the State
    virtual void SetOw (const int State, const int Ow) = 0;

};

}

#endif
