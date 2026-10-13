/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_ACTIONSA_H_
#define _FA_ACTIONSA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FABrResultCA;

///
/// Execution call-back for right part actions.
///

class FAActionsA {

public:
    virtual void Process (
        const int ActNum,           // RuleNum 0..MaxRule
        const int From,             // starting match position
        const int To,               // ending match position
        const FABrResultCA * pRes,  // sub-expressions
        void * pContext = NULL      // context to be modified
    ) const = 0;
};

}

#endif
