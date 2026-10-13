/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTMORPH_B2W_H_
#define _FA_TESTMORPH_B2W_H_

#include "FAConfig.h"
#include "FATestMorph.h"
#include "FAChain2Num_judy.h"

namespace BlingFire
{

///
/// B2W quality test
///
/// Input:
/// ...
/// Base\tWord1\tWord2\n
/// Base\n
/// ...
///
/// Note: Base may not be in { Word1, WordN } set
///

class FATestMorph_b2w : public FATestMorph {

public:
    FATestMorph_b2w (FAAllocatorA * pAlloc);

public:
    void Test (const char * pLineStr, const int LineLen);

private:
    const int Word2Id (const int * pWord, const int WordLen);

private:
    FAChain2Num_judy m_word2id;
    int m_MaxId;
};

}

#endif
