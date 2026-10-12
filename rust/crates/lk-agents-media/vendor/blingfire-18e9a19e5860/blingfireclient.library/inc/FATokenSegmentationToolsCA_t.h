/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TOKENSEGMENTATIONTOOLSCA_T_H_
#define _FA_TOKENSEGMENTATIONTOOLSCA_T_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// This is a common interface for different tokenization algorithms
///   to avoid having a many if/then/else at runtime.
///

template < class Ty >
class FATokenSegmentationToolsCA_t {

public:
    virtual const int Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize,
            const int UnkId
        ) const = 0;
};

}

#endif
