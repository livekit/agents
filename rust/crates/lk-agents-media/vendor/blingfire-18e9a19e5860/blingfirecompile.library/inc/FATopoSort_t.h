/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TOPOSORT_T_H_
#define _FA_TOPOSORT_T_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAAllocatorA.h"
#include "FAUtils.h"

namespace BlingFire
{

///
/// This processor calculates a topological order.
///
/// Input: G(V,E), |V| == N,
///
/// Output: array T, s.t. 
///   i < j <=> T[i] topologically greater T[j]
///   (exists path from T[i] to T[j])
///
/// Note: 
///   _TGraph should have two methods implemented:
///  1. _TGraph::GetNodeCount - returns the number of nodes
///  2. _TGraph::GetDstNodes - returns destination nodes
///

template < class _TGraph >
class FATopoSort_t {

public:

  FATopoSort_t (FAAllocatorA * pAlloc);

public:

  /// sets up graph object
  void SetGraph (const _TGraph * pGraph);
  /// makes proessing
  void Process ();
  /// returns the resulting array (the size of the array is equal to NodeCount)
  const int GetTopoOrder (const int ** ppTopoOrder) const;

private:

  /// specifies the number of nodes in the input graph
  inline void prepare (const int NodeCount);
  /// adds next node and all its destination nodes
  inline void update_in_power (const int * pDstNodes, 
                               const int Size);

private:

  /// graph to be sorted
  const _TGraph * m_pGraph;
  /// topologically sorted nodes
  FAArray_cont_t < int > m_order;
  /// mapping from Node into the number of incomming arcs
  FAArray_cont_t < int > m_in_power;
  /// stack of top-most nodes
  FAArray_cont_t < int > m_stack;
  /// tmp destination nodes storage
  FAArray_cont_t < int > m_tmp_nodes;

};


template < class _TGraph >
FATopoSort_t< _TGraph >::FATopoSort_t (FAAllocatorA * pAlloc) :
  m_pGraph (NULL)
{
  m_order.SetAllocator (pAlloc);
  m_order.Create ();

  m_in_power.SetAllocator (pAlloc);
  m_in_power.Create ();

  m_stack.SetAllocator (pAlloc);
  m_stack.Create ();

  m_tmp_nodes.SetAllocator (pAlloc);
  m_tmp_nodes.Create ();
}


template < class _TGraph >
void FATopoSort_t< _TGraph >::prepare (const int NodeCount)
{
  DebugLogAssert (m_stack.empty ());

  m_order.resize (0);

  // initialize newly added values
  m_in_power.resize (NodeCount);
  for (int i = 0; i < NodeCount; ++i) {
    m_in_power [i] = 0;
  }

  m_tmp_nodes.resize (NodeCount);
}


template < class _TGraph >
void FATopoSort_t< _TGraph >::update_in_power (const int * pDstNodes,
                                               const int Size)
{
  for (int i = 0; i < Size; ++i) {

    DebugLogAssert (NULL != pDstNodes);
    const int DstNode = pDstNodes [i];
    m_in_power [DstNode]++;
  }
}


template < class _TGraph >
void FATopoSort_t< _TGraph >::SetGraph (const _TGraph * pGraph)
{
    m_pGraph = pGraph;
}


template < class _TGraph >
void FATopoSort_t< _TGraph >::Process ()
{
    DebugLogAssert (NULL != m_pGraph);
    DebugLogAssert (m_stack.empty ());

    const int NodeCount = m_pGraph->GetNodeCount ();

    /// make ncessary rellocations
    prepare (NodeCount);

    int * pDstNodes = m_tmp_nodes.begin ();

    /// update in-powers
    int i;
    for (i = 0; i < NodeCount; i++) {

        const int Size = \
            m_pGraph->GetDstNodes (i, pDstNodes, m_tmp_nodes.size ());

        if (0 < Size && (unsigned int) Size > m_tmp_nodes.size ()) {

            m_tmp_nodes.resize (Size);
            pDstNodes = m_tmp_nodes.begin ();
            m_pGraph->GetDstNodes (i, pDstNodes, Size);
        }

        update_in_power (pDstNodes, Size);
    }

    /// put all nodes with in_power == 0 into stack
    for (i = 0; i < NodeCount; i++) {

        const int InPower = m_in_power [i];

        if (0 == InPower)
            m_stack.push_back (i);
    }

    /// calc the ordering
    while (!m_stack.empty ()) {

        const int Node = m_stack [m_stack.size () - 1];
        m_stack.pop_back ();

        m_order.push_back (Node);

        const int Size = \
            m_pGraph->GetDstNodes (Node, pDstNodes, m_tmp_nodes.size ());

        if (0 < Size && (unsigned int) Size > m_tmp_nodes.size ()) {

            m_tmp_nodes.resize (Size);
            pDstNodes = m_tmp_nodes.begin ();
            m_pGraph->GetDstNodes (Node, pDstNodes, Size);
        }

        for (i = 0; i < Size; ++i) {

            DebugLogAssert (pDstNodes);

            const int DstNode = pDstNodes [i];
            m_in_power [DstNode]--;

            if (0 == m_in_power [DstNode])
                m_stack.push_back (DstNode);
        }
    }
}


template < class _TGraph >
const int 
FATopoSort_t< _TGraph >::GetTopoOrder (const int ** ppTopoOrder) const
{
    DebugLogAssert (ppTopoOrder);

    *ppTopoOrder = m_order.begin ();

    return m_order.size ();
}

}

#endif
