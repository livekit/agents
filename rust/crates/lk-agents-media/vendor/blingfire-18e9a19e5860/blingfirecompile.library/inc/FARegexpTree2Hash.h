/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREE2HASH_H_
#define _FA_REGEXPTREE2HASH_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FATopoSort_t.h"
#include "FARegexpTreeTopoGraph.h"

namespace BlingFire
{

class FAAllocatorA;
class FARegexpTree;

///
/// This class calculates hash values for each node of the regular 
/// expression tree.
///

class FARegexpTree2Hash {

public:
    FARegexpTree2Hash (FAAllocatorA * pAlloc);

public:

    // sets up original regular expression text
    // offsets from pTree refer to this text
    void SetRegexp (const char * pRegexp, const int Length);
    // regular expression tree
    void SetRegexpTree (FARegexpTree * pTree);
    // builds hash values for each node
    void Process ();
    // returns hash value
    const int GetKey (const int NodeId) const;
    // updates NodeId's hash value and all the parents
    void Update (const int NodeId);

private:

    inline const int Node2Key (const int NodeId) const;
    inline const int Symbol2Key (const int Offset, const int Length) const;

private:

    const char * m_pRegexp;
    int m_Length;
    FARegexpTree * m_pTree;
    // topological sorter
    FATopoSort_t < FARegexpTreeTopoGraph > m_sort;
    // node -> key
    FAArray_cont_t < int > m_node2key;

};

}

#endif
