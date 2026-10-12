/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ACTIONSARRAYA_H_
#define _FA_ACTIONSARRAYA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAActionsA;

///
/// Common interface for the array of stages of actions.
///

class FAActionsArrayA {

public:
    /// returns stage count
    virtual const int GetStageCount () const = 0;
    /// returns i-th stage pointer, i = 0..(Count-1)
    virtual const FAActionsA * GetStage (const int Num) const = 0;
};

}

#endif
