/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREESIMPLIFY_DISJ_H_
#define _FA_REGEXPTREESIMPLIFY_DISJ_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FATopoSort_t.h"
#include "FARegexpTreeTopoGraph.h"
#include "FARegexpTree2Hash.h"
#include "FARegexpTree2Str.h"

namespace BlingFire
{

class FARegexpTree;
class FAAllocatorA;

///
/// This class simplifies disjunctions.
///
/// In reverse topological order do:
///   1. UniqDisj:
///   1.1 Disj(A, A) --> A
///   1.2 Disj(Disj(B, A), A) --> Disj(B, A)
///
///   2. Left factorization:
///   2.1. Disj(Conc(...Conc(P, A), B), Conc(...Conc(P, C), D)) -->
///        Conc(P, Disj(Conc(...A, B), Conc(...C, D)));
///   2.1. Disj(Disj(Q, Conc(...Conc(P, A), B)), Conc(...Conc(P, C), D)) -->
///        Disj(Q, Conc(P, Disj(Conc(...A, B), Conc(...C, D))));
///
/// Examples:
///   1. "(10|10|(10 10)|(10 10)) => (10)|(10 10)"
///   2. "(.* 10 20) | (.* 10 30)" => "(.* 10) (20|30)"
///

class FARegexpTreeSimplify_disj {

public:
    FARegexpTreeSimplify_disj (FAAllocatorA * pAlloc);

public:
    void SetRegexp (const char * pRegexp, const int Length);
    void SetRegexpTree (FARegexpTree * pTree);
    void Process ();

private:

    // calculates topological order of the nodes
    void TopoOrder ();

    // uniques disjunctions, assumed that they are sorted
    void UniqDisj ();
    // Disj(A, A) --> A
    inline void Simplify_11 (
            const int NodeId, 
            const int TrBr, 
            const int TrBrOffset
        );
    // Disj(Disj(B, A), A) --> Disj(B, A)
    inline void Simplify_12 (
            const int NodeId, 
            const int TrBr, 
            const int TrBrOffset
        );

    // puts common prefixes out of disjunctions
    void LeftFact ();
    // finds common prefix node in left sub-tree and right sub-tree
    // returns false if there were no common prefix found
    inline const bool FindCommonPref (
            const int LeftConc,
            const int RightConc,
            int * pLeftPref,
            int * pRightPref
        );
    // Disj(Conc(...Conc(P, A), B), Conc(...Conc(P, C), D)) --> 
    // Conc(P, Disj(Conc(...A, B), Conc(...C, D))),
    inline void Simplify_21 (
            const int NodeId,
            const int TrBr,
            const int TrBrOffset,
            const int LeftPref,
            const int RightPref
        );
    // Disj(Disj(Q, Conc(...Conc(P, A), B)), Conc(...Conc(P, C), D)) -->
    // Disj(Q, Conc(P, Disj(Conc(...A, B), Conc(...C, D))));
    inline void Simplify_22 (
            const int NodeId,
            const int DisjTrBr,
            const int DisjTrBrOffset,
            const int ConcTrBr,
            const int ConcTrBrOffset,
            const int LeftPref,
            const int RightPref
        );

private:
    // regexp tree
    FARegexpTree * m_pTree;
    // topological sorter
    FATopoSort_t < FARegexpTreeTopoGraph > m_sort;
    // hash for each node
    FARegexpTree2Hash m_node2key;
    // string representation for two arbitrary nodes
    FARegexpTree2Str m_Node1Str;
    FARegexpTree2Str m_Node2Str;

    const char * m_pTmpRegexp;
};

}

#endif
