/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREESORT_H_
#define _FA_REGEXPTREESORT_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FARegexpTree.h"
#include "FATopoSort_t.h"
#include "FARegexpTree2Hash.h"
#include "FABitArray.h"
#include "FARegexpTreeTopoGraph.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Sorts the arguments of disjunctions such that shortest arguments appear
/// in the left most positions. The algorithms works as follows:
///
/// 1. For each node of the tree Min and Max number of concatenation operators
///    is calculated
///
/// 2. For each group of the adjacent disjunction nodes with the same TrBr Id
///      sort all their children with the following less operator:
///
///      if (N1.Min < N2.Min)
///        return true;
///      else if (N1.Min == N2.Min && N1.Max < N2.Max)
///        return true;
///      else
///        return N1."Regexp" < N2."Regexp";
///
/// Examples:
///
/// 1. "(10 20 30)|(10)|(10 20)" --> "(10)|(10 20)|(10 20 30)"
/// 2. "(10? 20? 30?)|(10*)|(10+ 20)" --> "(10*)|(10+ 20)|(10? 20? 30?)"
/// 3. "(10 20 30)|(10 ((20)|(20 30)))" --> "(10 ((20)|(20 30)))|(10 20 30)"
///

class FARegexpTreeSort {

public:
    FARegexpTreeSort (FAAllocatorA * pAlloc);

public:
    // sets up regexp text
    void SetRegexp (const char * pRegexp, const int Length);
    // sets up regexp tree
    void SetRegexpTree (FARegexpTree * pTree);
    // sets up reverse sorting order (false by default)
    void SetReverse (const bool Reverse);
    // makes actual sorting
    void Process ();

private:
    // makes internal containers ready
    void Prepare ();
    // makes nodes topological sorting
    void TopoSort ();
    // calculates Min/Max mertics for each node
    void CalcMinMax ();
    // returns the top most parent for the cluster of adjacent disjunctions
    // with respect to TrBr Id
    const int FindTopMostParent (const int NodeId) const;
    // fills in m_disj_nodes and m_child_nodes 
    void GetAdjNodes (const int NodeId);
    // bulids a tree from m_disj_nodes and sorted m_child_nodes
    void ModifyTree ();
    // makes actual work - reorders disjuction's child nodes
    void Reorder ();
    // returns processor into the initial state
    void Clear ();

private:

    // takes length into account
    class _TNodeCmp {
    public:
        _TNodeCmp (
            const int * pNode2Min,
            const int * pNode2Max,
            const FARegexpTree2Hash * pRe2Hash,
            const bool Reverse
        );

    public:
        const bool operator () (const int NodeId1, const int NodeId2) const;

    private:
        const int * m_pNode2Min;
        const int * m_pNode2Max;
        const FARegexpTree2Hash * m_pRe2Hash;
        const bool m_Reverse;
    };

private:
    // regexp tree
    FARegexpTree * m_pTree;
    // topological sorter
    FATopoSort_t < FARegexpTreeTopoGraph > m_sort;

    // minimal number of concatenated terminals
    FAArray_cont_t < int > m_node2min;
    // maximum number of concatenated terminals
    FAArray_cont_t < int > m_node2max;
    // sorted disjuction nodes
    FABitArray m_seen;
    // adjacent disjunctions
    FAArray_cont_t < int > m_disj_nodes;
    // children of adjacent disjunctions
    FAArray_cont_t < int > m_child_nodes;
    // temporary stack
    FAArray_cont_t < int > m_stack;
    // keeps hash values for all the nodes
    FARegexpTree2Hash m_re2hash;
    // sorting order
    bool m_Reverse;
};

}

#endif
