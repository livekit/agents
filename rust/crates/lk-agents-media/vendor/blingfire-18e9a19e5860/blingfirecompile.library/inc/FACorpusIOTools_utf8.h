/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CORPUSIOTOOLS_UTF8_H_
#define _FA_CORPUSIOTOOLS_UTF8_H_

#include "FAConfig.h"

#include <iostream>

namespace BlingFire
{

class FAAllocatorA;
class FATaggedTextA;
class FATaggedTextCA;
class FAParseTreeA;
class FATagSet;

///
/// Reads/Writes different objects from textual representation.
///

class FACorpusIOTools_utf8 {

public:
    FACorpusIOTools_utf8 (FAAllocatorA * pAlloc);

public:
    /// sets up tagset for both tree nodes and words
    void SetTagSet (const FATagSet * pTagSet);
    /// specifies whether words don't have POS tags
    void SetNoPosTags (const bool NoPosTags);

public:
    // reads tagged text object
    void Read (
            std::istream& is, 
            FATaggedTextA * pS
        ) const;
    // writes tagged text object
    void Print (
            std::ostream& os, 
            const FATaggedTextCA * pS
        ) const;

    // reads parsed text object
    void Read (
            std::istream& is, 
            FATaggedTextA * pS, 
            FAParseTreeA * pT
        ) const;
    // writes parsed text object
    void Print (
            std::ostream& os, 
            const FATaggedTextCA * pS, 
            const FAParseTreeA * pT
        ) const;

private:
    FAAllocatorA * m_pAlloc;
    const FATagSet * m_pTagSet;
    bool m_NoPosTags;
};

}

#endif
