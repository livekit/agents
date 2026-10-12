/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MERGE_SETS_H_
#define _FA_MERGE_SETS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Creates a system of non-intersecting sets by merging pairs of sets. 
/// Can be used to find connected components.
///

class FAMergeSets {

public:
    FAMergeSets (FAAllocatorA * pAlloc);

public:
    // makes the object ready, creates Count sets of one element each
    void Prepare (const int Count);
    // merges two sets by two elements
    void Merge (const int e1, const int e2);
    // returns set idx for the element e
    const int GetSet (const int E);

private:
    /// faster m_e2p access
    int * m_pE2P;
    /// maps e -> p
    /// if p > 0 then p is parent index + 1
    /// if p <= 0 then e is the root of a set and |p| is a rank value
    FAArray_cont_t < int > m_e2p;
    /// stack for path compression
    FAArray_cont_t < int > m_stack;
};

}

#endif
