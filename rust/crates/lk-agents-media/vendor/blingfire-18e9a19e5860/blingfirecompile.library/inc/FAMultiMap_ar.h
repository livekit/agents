/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTI_MAP_AR_H_
#define _FA_MULTI_MAP_AR_H_

#include "FAConfig.h"
#include "FAMultiMapA.h"
#include "FAArray_cont_t.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Array-based implementation of FAMultiMapA.
///
/// Note:
///   1. Use this implementation when mapping is totally defined.
///   2. Cannot hold "undefined" value, if Key was inserted then mapping
///      is defined for all Key_i <= Key.
///

class FAMultiMap_ar : public FAMultiMapA {

public:

    FAMultiMap_ar ();
    virtual ~FAMultiMap_ar ();

public:

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
    void Add (const int Key, const int Value);
    const int Next (int * pKey, const int ** ppValues) const;
    const int Prev (int * pKey, const int ** ppValues) const;

/// additional functionality
public:

    // makes Arrays of values sorted and uniq
    void SortUniq ();

private:
    // makes sure that m_x2ys [X] contains new empty entry
    inline void ensure (const int Key);

private:

    FAArray_cont_t < FAArray_cont_t < int > >  m_key2vals;
    int m_MaxCount;
    FAAllocatorA * m_pAlloc;

};

}

#endif
