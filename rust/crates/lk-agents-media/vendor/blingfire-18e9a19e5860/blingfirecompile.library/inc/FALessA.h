/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_LESSA_H_
#define _FA_LESSA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FALessA {

public:
    virtual const bool Less (const int Val1, const int Val2) const = 0;

};

}

#endif
