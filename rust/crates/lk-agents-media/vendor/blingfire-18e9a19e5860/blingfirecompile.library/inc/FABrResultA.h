/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_BRRESULTA_H_
#define _FA_BRRESULTA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Interface for setting up results of extracting brackets
///

class FABrResultA {

public:
    // adds new result
    virtual void AddRes (const int BrId, const int From, const int To) = 0;
};

}

#endif
