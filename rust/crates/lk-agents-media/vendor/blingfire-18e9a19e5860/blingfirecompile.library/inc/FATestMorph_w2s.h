/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTMORPH_W2S_H_
#define _FA_TESTMORPH_W2S_H_

#include "FAConfig.h"
#include "FATestMorph.h"

namespace BlingFire
{

///
/// W2S quality test
///
/// Input:
/// ...\n
/// Word1Word2...WordN\tWord1\tWord2\t...WordN\n
/// ...\n
///

class FATestMorph_w2s : public FATestMorph {

public:
    FATestMorph_w2s (FAAllocatorA * pAlloc);

public:
    void Test (const char * pLineStr, const int LineLen);

private:
    inline const int BuildInputSegs (
            const int * pInBuff, 
            const int InBuffSize
        );
    /// returns true if splitting "looks good",
    /// bad one can be generated from inconsistent test data
    inline const bool IsValid (const int * pB, const int BSize) const;

};

}

#endif
