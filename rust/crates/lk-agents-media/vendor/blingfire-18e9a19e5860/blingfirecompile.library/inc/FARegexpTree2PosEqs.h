/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREE2POSEQS_H_
#define _FA_REGEXPTREE2POSEQS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAMap_judy.h"
#include "FATopoSort_t.h"
#include "FABitArray.h"
#include "FARegexpTreeTopoGraph.h"

namespace BlingFire
{

class FARegexpTree;

///
/// This class calculates equivalence classes on regular expresion's
/// positions.
///
/// Note:
///
/// 1. In order to get maximum gain from the class regular expression is
///    ought to be simplified, e.g. (a*b)* -> (a|b)*; (a b d)|(a c d) -> 
///    a(b|c)d; (a+b?)* -> (a|b)* ...
///
/// 2. All disjunctions in sub-expressions have to be left or right 
///    associative only.
///

class FARegexpTree2PosEqs {

public:
    FARegexpTree2PosEqs (FAAllocatorA * pAlloc);

public:
    // sets up regular expression tree
    void SetRegexpTree (const FARegexpTree * pTree);
    // calculates position equivalence classes
    void Process ();
    // returns mapping from each position to the smallest equivalent position
    const FAMapA * GetPos2Class () const;

private:
    /// builds node -> pos map
    void CalcNode2Pos ();
    /// sorts nodes in topological order
    void TopoSort ();
    /// finds a set of equivalent positions for the given node NodeId
    void FindEqPos (const int NodeId);
    /// fills m_Pos2Class in
    void BuildEqClasses ();

private:
    // input tree
    const FARegexpTree * m_pTree;
    // topological sorter
    FATopoSort_t < FARegexpTreeTopoGraph > m_sort;
    // pos: 0 .. m_MaxPos
    int m_MaxPos;
    // mapping from node_id into position, -1 if undefined
    FAArray_cont_t < int > m_Node2Pos;
    // temporary storage of equivalent nodes and positions
    FAArray_cont_t < int > m_eq_pos;
    // seen nodes
    FABitArray m_seen;
    // mapping from pos to the smallest equivalent position
    FAMap_judy m_Pos2Class;
};

}

#endif
