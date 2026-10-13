/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TESTMORPH_TAG_DICT_H_
#define _FA_TESTMORPH_TAG_DICT_H_

#include "FAConfig.h"
#include "FATestMorph.h"
#include "FADictInterpreter_t.h"

namespace BlingFire
{

class FATagSet;

///
/// Tag Dict quality test
///
/// Input:
/// ...
/// word\tTag1\tTag2\t...TagN\n
/// ...
///

class FATestMorph_tag_dict : public FATestMorph {

public:
    FATestMorph_tag_dict (FAAllocatorA * pAlloc);

public:
    void SetTagDict (const FADictInterpreter_t < int > * pDict);
    void SetTagSet (const FATagSet * pTagSet);
    void Test (const char * pLineStr, const int LineLen);

private:
    const FADictInterpreter_t < int > * m_pDict;
    const FATagSet * m_pTagSet;
};

}

#endif
