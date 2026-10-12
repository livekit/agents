/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ACT2ARRA_H_
#define _FA_ACT2ARRA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FATagSet;

///
/// a callback interface for converting an action text into the array 
/// of integers that runtime can use
///

class FAAct2ArrA {

public:
    /// tagset (symbol table) setup
    virtual void SetTagSet (const FATagSet * pTagSet) = 0;

    /// returns array size (-1 in case of a syntax error, 0 in case 
    /// of an empty action)
    virtual const int Process (
            const char * pStr, 
            const int Len,
            const int ** ppArr,
            int * pLeftCxAdjust,
            int * pRightCxAdjust
        ) = 0;
};

}

#endif
