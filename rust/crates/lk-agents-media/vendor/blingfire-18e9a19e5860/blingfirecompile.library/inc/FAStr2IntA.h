/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STR2INTA_H_
#define _FA_STR2INTA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAStr2IntA {

public:
    /// finds value by the string
    virtual const bool Get (const char * pStr, 
                            const int Size, 
                            int * pValue) const = 0;
    /// returns the total number of pairs stored in map
    virtual const int GetStrCount () const = 0;
    /// returns String by its idx
    ///   idx \in [0, GetChainCount ())
    ///   return value is the size of the Chain
    virtual const int GetStr (const int Idx,
                              const char ** ppStr) const = 0;
    /// returns value by its idx
    virtual const int GetValue (const int Idx) const = 0;

    /// adds String -> Value pair, returns index
    virtual const int Add (const char * pStr,
                           const int Size,
                           const int Value) = 0;
};

}

#endif
