/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ACT2ARR_CSS_H_
#define _FA_ACT2ARR_CSS_H_

#include "FAConfig.h"
#include "FAAct2ArrA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FATagSet;

///
/// Parses CSS suggestion rules of the following format:
///
///  WORD [ WORD ] SCORE [ EDIT_TYPE ]
///  WORD ::= "any text" (only quotes should be escaped e.g. "\"")
///  SCORE ::= a C-style floating-point number in the range:
///            from -INT_MAX/127 to +INT_MAX/127
///  EDIT_TYPE ::= one of edit type tags
///   (where [X] means X is optional)
///

class FAAct2Arr_css : public FAAct2ArrA {

public:
    FAAct2Arr_css (FAAllocatorA * pAlloc);
    
public:
    void SetTagSet (const FATagSet * pTagSet);
    /// (see FAAct2ArrA for details)
    const int Process (const char * pStr, const int Len, const int ** ppArr, 
        int * pLeftCxAdjust, int * pRightCxAdjust);

private:
    inline const int ParseWord (const char * pStr, const int Len);
    inline const int ParseScore (const char * pStr, const int Len);
    inline const int ParseEditType (const char * pStr, const int Len);

private:
    FAArray_cont_t < int > m_arr;
    const FATagSet * m_pTagSet;
};

}

#endif
