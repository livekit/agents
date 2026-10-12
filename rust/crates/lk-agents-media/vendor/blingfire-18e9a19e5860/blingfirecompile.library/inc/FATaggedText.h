/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TAGGEDTEXT_H_
#define _FA_TAGGEDTEXT_H_

#include "FAConfig.h"
#include "FATaggedTextA.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Tagged text container.
///

class FATaggedText : public FATaggedTextA {

public:
    FATaggedText (FAAllocatorA * pAlloc);
    virtual ~FATaggedText ();

public:
    const int GetWordCount () const;
    const int GetWord (const int Num, const int ** pWord) const;
    const int GetTag (const int Num) const;
    const int GetOffset (const int Num) const;

public:
    void AddWord (
            const int * pWord,
            const int Length,
            const int Tag
        );
    void AddWord (
            const int * pWord,
            const int Length,
            const int Tag,
            const int Offset
        );

    void Clear ();

public:
    void SetTags (const int * pTags, const int Count);

private:
    FAArray_cont_t < int > m_chars;
    FAArray_cont_t < int > m_num2sl;
    FAArray_cont_t < int > m_num2tag;
    FAArray_cont_t < int > m_num2offset;
};

}

#endif
