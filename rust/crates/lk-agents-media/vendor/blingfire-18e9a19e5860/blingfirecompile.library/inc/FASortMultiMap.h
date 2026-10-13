/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_SORTMULTIMAP_H_
#define _FA_SORTMULTIMAP_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAMultiMapA;
class FAAllocatorA;

///
/// Sorts multimap keys with respect to the lexicographic order of the 
/// associated arrays of values.
///

class FASortMultiMap {

public:
    FASortMultiMap (FAAllocatorA * pAlloc);

public:
    /// sets up input multi-map
    void SetMultiMap (const FAMultiMapA * pMap);
    /// sets up direction, e.g. l2r, r2l in which to compare arrays of values
    void SetDirection (const int Direction);
    /// sorts the chains
    void Process ();
    /// returns ordered keys
    const int GetKeyOrder (const int ** ppKeys) const;

private:
    // Key1 < Key2 if Chain[Key1] < Chain[Key2] lexicographically
    class _TKeyCmp_l2r {
    public:
        _TKeyCmp_l2r (const FAMultiMapA * pMap);

    public:
        const bool operator () (const int Key1, const int Key2) const;

    private:
        const FAMultiMapA * m_pMap;
    };

    // Key1 < Key2 if Rev(Chain[Key1]) < Rev(Chain[Key2]) lexicographically,
    //   where Rev([a1, a2, ... an]) == [an, ..., a2, a1]
    class _TKeyCmp_r2l {
    public:
        _TKeyCmp_r2l (const FAMultiMapA * pMap);

    public:
        const bool operator () (const int Key1, const int Key2) const;

    private:
        const FAMultiMapA * m_pMap;
    };

private:
    const FAMultiMapA * m_pMap;
    FAArray_cont_t < int > m_keys;
    int m_Direction;
};

}

#endif
