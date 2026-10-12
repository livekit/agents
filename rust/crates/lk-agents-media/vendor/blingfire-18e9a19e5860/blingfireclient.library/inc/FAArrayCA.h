/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ARRAYCA_H_
#define _FA_ARRAYCA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// This is a common interface for packed read-only arrays.
///

class FAArrayCA {

public:
    /// returns value by the index, 0..Count-1
    virtual const int GetAt (const int Idx) const = 0;
    /// returns number of elementes in the array
    virtual const int GetCount () const = 0;
};

}

#endif
