/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_TRANSCLOSURE_ACYC_T_H_
#define _FA_TRANSCLOSURE_ACYC_T_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FATopoSort_t.h"
#include "FASetUtils.h"
#include "FAAllocatorA.h"

namespace BlingFire
{

///
/// This class calculates transistive closure for acyclic graphs.
///

template < class _TGraph >
class FATransClosure_acyc_t {

public:

  FATransClosure_acyc_t (FAAllocatorA * pAlloc);

public:

  void SetInGraph (const _TGraph * pInGraph);
  void SetOutGraph (_TGraph * pOutGraph);
  void Process ();

private:

  void ProcessNode (const int Node);

private:

  /// input relation
  const _TGraph * m_pInGraph;
  /// output relation
  _TGraph * m_pOutGraph;
  /// sets to be merged
  FAArray_cont_t < const int * > m_set_ptrs;
  FAArray_cont_t < int > m_set_lengths;
  /// topological sorter
  FATopoSort_t < _TGraph > m_sorter;
  /// set operations
  FASetUtils m_set_ops;

};


template < class _TGraph >
FATransClosure_acyc_t< _TGraph >::
FATransClosure_acyc_t (FAAllocatorA * pAlloc) :
  m_sorter (pAlloc),
  m_set_ops (pAlloc)
{
  m_set_ptrs.SetAllocator (pAlloc);
  m_set_ptrs.Create ();

  m_set_lengths.SetAllocator (pAlloc);
  m_set_lengths.Create ();
}


template < class _TGraph >
void FATransClosure_acyc_t< _TGraph >::SetInGraph (const _TGraph * pInGraph)
{
  m_pInGraph = pInGraph;

  m_sorter.SetGraph (pInGraph);
}


template < class _TGraph >
void FATransClosure_acyc_t< _TGraph >::SetOutGraph (_TGraph * pOutGraph)
{
  m_pOutGraph = pOutGraph;
}


template < class _TGraph >
void FATransClosure_acyc_t< _TGraph >::Process ()
{
  DebugLogAssert (m_pInGraph);
  DebugLogAssert (m_pOutGraph);
  
  // calc topological order
  m_sorter.Process ();

  const int * pOrder;
  const int NodeCount = m_sorter.GetTopoOrder (&pOrder);

  // make processing in the reverse topological order

  for (int i = 1; i <= NodeCount; ++i) {

    DebugLogAssert (NULL != pOrder);

    const int Node = pOrder [NodeCount - i];
    ProcessNode (Node);
  }
}


template < class _TGraph >
void FATransClosure_acyc_t< _TGraph >::ProcessNode (const int Node)
{
  m_set_ptrs.resize (0);
  m_set_lengths.resize (0);

  // get the Node's destination set
  const int * pDstNodes;
  const int DstSize = m_pInGraph->GetDstNodes2 (Node, &pDstNodes);

  if (0 < DstSize) {

    DebugLogAssert (pDstNodes);

    m_set_ptrs.push_back (pDstNodes);
    m_set_lengths.push_back (DstSize);

    for (int i = 0; i < DstSize; ++i) {

        // get destination node 
        const int DstNode = pDstNodes [i];

        // get the closure value for this node
        // (as the graph is topologically sorted the node is already processed)
        const int * pNodes;
        const int Size = m_pOutGraph->GetDstNodes2 (DstNode, &pNodes);

        if (0 < Size) {
            m_set_ptrs.push_back (pNodes);
            m_set_lengths.push_back (Size);
        }
    }

  } // of if

  DebugLogAssert (m_set_lengths.size () == m_set_ptrs.size ());

  // calc the union
  m_set_ops.UnionN (m_set_ptrs.begin (),
                    m_set_lengths.begin (),
                    m_set_lengths.size (),
                    2);

  // get the result
  const int * pRes;
  const int Size = m_set_ops.GetRes (&pRes, 2);

  // set up the result
  if (0 < Size) {
      m_pOutGraph->SetDstNodes (Node, pRes, Size);
  }
}

}

#endif
