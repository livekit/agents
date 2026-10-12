/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTMORPH_W2T_H_
#define _FA_TESTMORPH_W2T_H_

#include "FAConfig.h"
#include "FATestMorph.h"

namespace BlingFire
{

class FATagSet;

///
/// W2T quality test
///
/// Input:
/// ...
/// word\tTag1\tTag2\t...TagN\n
/// ...
///

class FATestMorph_w2t : public FATestMorph {

public:
    FATestMorph_w2t (FAAllocatorA * pAlloc);

public:
    // specifies which function to test, W2T or B2T (W2T is default)
    void SetFunc (const int Func);
    // sets up the tagset
    void SetTagSet (const FATagSet * pTagSet);
    void Test (const char * pLineStr, const int LineLen);

private:
    int m_Func;
    const FATagSet * m_pTagSet;
};

}

#endif
