/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_BRRESULTCA_H_
#define _FA_BRRESULTCA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Interface for reading results of extracting brackets
///

class FABrResultCA {

public:
    /// returns an array of all added BrId --> [<From, To>, ... , <From, To>]
    /// (returns -1 if no BrId bracket was extracted)
    virtual const int GetRes (const int BrId, const int ** ppData) const = 0;
    /// returns next result BrId --> [<From, To>, ... , <From, To>]
    /// (the initial value of BrId should be -1)
    /// (returns -1 if no BrId bracket was extracted)
    virtual const int GetNextRes (int * pBrId, const int ** ppData) const = 0;
    /// returns first added BrId --> <From, To> relation
    /// (returns -1 if no BrId bracket was extracted)
    virtual const int GetFrom (const int BrId) const = 0;
    virtual const int GetTo (const int BrId) const = 0;

};

}

#endif
