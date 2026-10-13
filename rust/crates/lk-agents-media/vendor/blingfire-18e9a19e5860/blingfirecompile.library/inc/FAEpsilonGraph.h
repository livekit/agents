/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_EPSILON_GRAPH_H_
#define _FA_EPSILON_GRAPH_H_

#include "FAConfig.h"
#include "FASetUtils.h"
#include "FAArray_cont_t.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;
class FARSNfaA;


///
/// This class implementes sub graph interface for FARSNfaA class over
/// the epsilon transition.
///
/// Notes:
///
///  1. FAEpsilonGraph::SetDstNodes converts epsilon transitions into 
///     non-epsilon transitions.
///  2. Used by both FATopoSort_t and FATransClosure_acyc_t classes
///

class FAEpsilonGraph {

public:
    FAEpsilonGraph (FAAllocatorA * pAlloc);
    virtual ~FAEpsilonGraph ();

public:
    /// sets up nfa for reading
    void SetInNfa (const FARSNfaA * pInNfa);
    /// sets up nfa for writing
    void SetOutNfa (FARSNfaA * pOutNfa);
    /// specifies the Epsilon symbol
    void SetEpsilonIw (const int EpsilonIw);

public:
    /// returns the number of nodes in graph, is used by FATopoSort_t class
    virtual const int GetNodeCount () const;
    /// returns real size of the set of destination nodes, 
    /// Note: pDstNodes should be allocated or NULL if only size is required
    virtual const int GetDstNodes (
            const int Node,
            __out_ecount(DstNodesSize) int * pDstNodes,
            const int DstNodesSize
        ) const;
    /// 
    virtual const int GetDstNodes2 (
            const int Node, 
            const int ** ppDstNodes
        ) const;
    /// sets up the set of destination states, prev version if any should be removed
    virtual void SetDstNodes (
            const int Node, 
            const int * pDstNodes, 
            const int Size
        );

protected:

    const FARSNfaA * m_pInNfa;
    FARSNfaA * m_pOutNfa;
    int m_EpsilonIw;
    /// sets to be merged
    FAArray_cont_t < const int * > m_set_ptrs;
    FAArray_cont_t < int > m_set_lengths;
    FASetUtils m_set_ops;
    /// dead state set keeper
    const int m_DeadState;

};

}

#endif
