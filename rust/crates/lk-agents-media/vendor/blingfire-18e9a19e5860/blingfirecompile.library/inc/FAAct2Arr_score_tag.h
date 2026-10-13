/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ACT2ARR_SCORE_TAG_H_
#define _FA_ACT2ARR_SCORE_TAG_H_

#include "FAConfig.h"
#include "FAAct2ArrA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FATagSet;

///
/// Parses SCORE TAG actions of the following format:
///
///  SCORE TAG
///  SCORE ::= an integer value
///  TAG ::= one of tags defined in the tagset
///

class FAAct2Arr_score_tag : public FAAct2ArrA {

public:
    FAAct2Arr_score_tag (FAAllocatorA * pAlloc);

public:
    void SetTagSet (const FATagSet * pTagSet);
    /// (see FAAct2ArrA for details)
    const int Process (const char * pStr, const int Len, const int ** ppArr, 
        int * pLeftCxAdjust, int * pRightCxAdjust);

private:
    inline const int ParseScore (const char * pStr, const int Len);
    inline const int ParseTag (const char * pStr, const int Len);

private:
    FAArray_cont_t < int > m_arr;
    const FATagSet * m_pTagSet;
};

}

#endif
