/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STR2INT_HASH_H_
#define _FA_STR2INT_HASH_H_

#include "FAConfig.h"
#include "FAStr2IntA.h"
#include "FAMap_judy.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Maximum number of strings to be stored is (MAX_INT / 2) - 1
///

class FAStr2Int_hash : public FAStr2IntA {

public:
    FAStr2Int_hash (FAAllocatorA * pAlloc);
    virtual ~FAStr2Int_hash ();

public:
    const bool Get (const char * pStr, const int Size, int * pValue) const;
    const int GetStrCount () const;
    const int GetStr (const int Idx, const char ** ppStr) const;
    const int GetValue (const int Idx) const;
    const int Add (const char * pStr, const int Size, const int Value);

public:
    void Clear ();

private:
    // generates hash-key for a given string
    inline static const int Str2Key (const char * pStr, const int Size);
    // identifies whether i-th string equals pStr
    inline const bool Equal (
            const int i, 
            const char * pStr, 
            const int Size
        ) const;
    // finds index by string or returns -1
    inline const int Str2Idx (const char * pStr, const int Size) const;
    // associates string, value pair with a new idx
    inline void PushBack (const char * pStr, const int Size, const int Value);
    // adds one more collition set
    inline void PushBackCSet ();

private:
    // Hash key to index mapping:
    // if idx > 0 then 
    //   i = idx - 1;
    // else if idx < 0 then
    //   j = (-idx) - 1;
    // else
    //   DebugLogAssert (0);
    FAMap_judy m_key2idx;
    // Storage of pairs:
    // i --> str
    FAArray_t < FAArray_cont_t < char > > m_i2str;
    // i --> value
    FAArray_t < int > m_i2value;
    // Storage of collision sets:
    // j --> { i_0, i_2, ... , i_n }
    FAArray_t < FAArray_cont_t < int > > m_csets;
    // allocator
    FAAllocatorA * m_pAlloc;
};

}

#endif
