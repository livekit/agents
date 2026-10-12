/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTMORPH_W2B_H_
#define _FA_TESTMORPH_W2B_H_

#include "FAConfig.h"
#include "FATestMorph.h"
#include "FAChain2Num_judy.h"

namespace BlingFire
{

///
/// W2B quality test
///
/// Input:
/// ...
/// Word1\tBase1\tBase2\n
/// Word2\tBase1\n
/// Word3\tBase2\n
/// ...
///

class FATestMorph_w2b : public FATestMorph {

public:
    FATestMorph_w2b (FAAllocatorA * pAlloc);

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
