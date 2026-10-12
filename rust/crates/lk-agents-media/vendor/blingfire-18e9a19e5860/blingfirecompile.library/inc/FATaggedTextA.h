/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TAGGEDTEXTA_H_
#define _FA_TAGGEDTEXTA_H_

#include "FAConfig.h"
#include "FATaggedTextCA.h"

namespace BlingFire
{

class FATaggedTextA : public FATaggedTextCA {

public:
    // adds a new word (no offset)
    virtual void AddWord (
            const int * pWord, 
            const int Length, 
            const int Tag
        ) = 0;
    // adds a new word with offset
    virtual void AddWord (
            const int * pWord, 
            const int Length, 
            const int Tag, 
            const int Offset
        ) = 0;
    // makes object empty
    virtual void Clear () = 0;
};

}

#endif
