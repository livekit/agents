/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORMCA_T_H_
#define _FA_TRANSFORMCA_T_H_

#include "FAConfig.h"
#include "FASecurity.h"

namespace BlingFire
{

template < class Ty >
class FATransformCA_t {

public:
    virtual const int Process (
            const Ty * pIn,
            const int InCount,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const = 0;
};

}

#endif
