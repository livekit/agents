/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TAGSET_H_
#define _FA_TAGSET_H_

#include "FAConfig.h"
#include "FAStr2Int_hash.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Tagset container.
///

class FATagSet : public FAStr2Int_hash {

public:
    FATagSet (FAAllocatorA * pAlloc);

public:
    // returns -1 if does not exist
    const int Tag2Str (const int Tag, const char ** ppStr) const;
    // returns -1 if does not exist
    const int Str2Tag (const char * pStr, const int Size) const;
    // overriden from FAStr2IntA
    const int Add (const char * pStr, const int Size, const int Tag);
    // Largest tag ID we've seen
    const int GetMaxTag () const;

private:
    inline void ensure (const int Tag);

private:
    // mapping from Tag to Str -> Value pair in FAStr2Int_hash
    FAArray_cont_t < int > m_tag2idx;
    int m_iMaxTag;
};

}

#endif
