/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAP_AR_UNIQ_H_
#define _FA_MULTIMAP_AR_UNIQ_H_

#include "FAConfig.h"
#include "FAMultiMapA.h"
#include "FAArray_cont_t.h"
#include "FAMap_judy.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Array-based implementation of FAMultiMapA.
///
/// Note:
///   1. Add (one element to the end) and Remove operations are not supported
///

class FAMultiMap_ar_uniq : public FAMultiMapA {

public:
    FAMultiMap_ar_uniq ();
    virtual ~FAMultiMap_ar_uniq ();

public:
    // sets up allocator (call before any usage)
    void SetAllocator (FAAllocatorA * pAlloc);
    // makes map as if it was just constructed
    void Clear ();

/// FAMultiMapA
public:
    const int Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const;
    const int GetMaxCount () const;

    const int Get (const int Key, const int ** ppValues) const;
    void Set (const int Key, const int * pValues, const int ValuesCount);
    // not implemented
    void Add (const int Key, const int Value);
    const int Next (int * pKey, const int ** ppValues) const;
    const int Prev (int * pKey, const int ** ppValues) const;

public:
    // makes Arrays of values sorted and uniq
    void SortUniq ();

private:
    // ensures m_key2idx holds the Key
    inline void ensure (const int Key);
    // calculates hash key
    inline static const int hash_key (const int * pValues, const int Size);
    // compares values with array by index Idx
    inline const bool equal (
            const int Idx, 
            const int * pValues, 
            const int Size
        ) const;

private:
    // mapping: key -> idx
    FAArray_cont_t < int >  m_key2idx;
    // mapping: idx -> array
    FAArray_cont_t < FAArray_cont_t < int > > m_idx2arr;
    // mapping: HashKey (array) -> idx
    FAMap_judy m_hash2idx;
    // keeps max number of elements Get can return
    int m_MaxCount;
    // allocator
    FAAllocatorA * m_pAlloc;
};

}

#endif
