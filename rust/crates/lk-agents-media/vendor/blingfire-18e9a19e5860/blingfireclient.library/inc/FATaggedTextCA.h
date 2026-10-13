/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_TAGGEDTEXTCA_H_
#define _FA_TAGGEDTEXTCA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FATaggedTextCA {

public:
    // returns number of words
    virtual const int GetWordCount () const = 0;
    // returns word text
    virtual const int GetWord (const int Num, const int ** pWord) const = 0;
    // returns POS tag
    virtual const int GetTag (const int Num) const = 0;
    // returns offset in the original text, -1 if not specified
    virtual const int GetOffset (const int Num) const = 0;
};

}

#endif
