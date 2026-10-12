/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTI_MAP_JUDY_H_
#define _FA_MULTI_MAP_JUDY_H_

#include "FAConfig.h"
#include "FAMultiMapA.h"
#include "FAMap_judy.h"
#include "FAArray_cont_t.h"
#include "FAHeap_t.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;

/// Judy-based implementation of the FAMultiMapA
/// see FAMultiMapA.h for details

class FAMultiMap_judy : public FAMultiMapA {

public:
    FAMultiMap_judy ();
    virtual ~FAMultiMap_judy ();

public:
    void SetAllocator (FAAllocatorA * pAlloc);

    const int Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const;
    const int GetMaxCount () const;

    const int Get (const int Key, const int ** ppValues) const;
    void Set (const int Key, const int * pValues, const int ValuesCount);
    void Add (const int Key, const int Value);
    const int Next (int * pKey, const int ** ppValues) const;
    const int Prev (int * pKey, const int ** ppValues) const;

/// additional functionality
public:
    /// makes Arrays of values sorted and uniq
    void SortUniq ();
    /// removes Key -> Values pair,
    /// restructures the map so pointers returned becomes invalid!
    void Remove (const int Key);
    /// makes map as if it was just constructed
    void Clear ();

private:
    inline const int GetNewIdx ();

private:
    /// map: key -> idx
    FAMap_judy m_key2idx;
    /// map: idx -> vals 
    FAArray_cont_t < FAArray_cont_t < int > > m_idx2vals;
    /// keeps unused indices
    FAHeap_t < int > m_deleted;
    /// allocator pointer
    FAAllocatorA * m_pAlloc;
    /// maximum size of the array asociated with the key
    int m_MaxCount;
};

}

#endif
