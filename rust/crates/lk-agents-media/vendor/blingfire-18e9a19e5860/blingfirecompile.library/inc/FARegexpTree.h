/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */



#ifndef _FA_REGEXPTREE_H_
#define _FA_REGEXPTREE_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FAHeap_t.h"

namespace BlingFire
{

class FARegexpTree {

public:

  FARegexpTree (FAAllocatorA* pAlloc);

public:

  /// not all of these token types will appear in the tree
  /// !!! BEWARE !!!
  /// These values must be synced with one used in FARegexpParser_msyacc
  enum {
    TYPE_OFFSET = 256,
    TYPE_SYMBOL              = 257,
    TYPE_ANY                 = 258,
    TYPE_LBR                 = 259,
    TYPE_RBR                 = 260,
    TYPE_CONCAT              = 261,
    TYPE_DISJUNCTION         = 262,
    TYPE_ITERATION           = 263,
    TYPE_NON_EMPTY_ITERATION = 264,
    TYPE_OPTIONAL            = 265,
    TYPE_BOUND_LBR           = 266,
    TYPE_COMMA               = 267,
    TYPE_BOUND_RBR           = 268,
    TYPE_EPSILON             = 269,
    TYPE_INTERVAL_LBR        = 270,
    TYPE_INTERVAL_RBR        = 271,
    TYPE_L_ANCHOR            = 272,
    TYPE_R_ANCHOR            = 273,
    TYPE_LTRBR               = 274,
    TYPE_RTRBR               = 275,
    TYPE_COUNT = TYPE_R_ANCHOR - TYPE_OFFSET
  };

public:

  /// returns MaxNodeId
  /// returns -1, if there are no nodes
  const int GetMaxNodeId () const;

  /// returns root node id
  const int GetRoot () const;
  /// sets up root
  void SetRoot (const int NodeId);

  /// returns node parent, returns -1 if there is no parent
  const int GetParent (const int NodeId) const;
  void SetParent (const int NodeId, const int ParentNodeId);
  /// returns left child
  const int GetLeft (const int NodeId) const;
  void SetLeft (const int NodeId, const int LeftNodeId);
  /// returns right child
  const int GetRight (const int NodeId) const;
  void SetRight (const int NodeId, const int RightNodeId);

  /// returns TrBrId, returns -1 if there are not trbr in this node
  const int GetTrBr (const int NodeId) const;
  void SetTrBr (const int NodeId, const int TrBr);
  /// returns offset of triangular bracket or -1 if does not exist
  const int GetTrBrOffset (const int NodeId) const;
  void SetTrBrOffset (const int NodeId, const int Offset);

  /// returns node type
  const int GetType (const int NodeId) const;
  /// returns node's label offset, -1 if there is no offset
  const int GetOffset (const int NodeId) const;
  /// returns node's label length, -1 if there is no length
  const int GetLength (const int NodeId) const;

  /// appends a new node to the tree, all the rest is initialized to -1
  /// returns node_id
  const int AddNode (const int NodeType, const int Offset, const int Length);
  /// returns whether NodeId is deleted
  const bool IsDeleted (const int NodeId) const;
  /// deletes node
  void DeleteNode (const int NodeId);
  /// deletes sub-tree
  void DeleteTree (const int NodeId);
  /// copies data from FromNode to ToNode, both nodes must exist
  /// does not links: Left, Right and Parent
  void CopyData (const int ToNodeId, const int FromNodeId);
  /// returns object into an initial state
  void Clear ();

private:

  /// returns [type, offset, length, left, right, trbr]
  inline int * GetNodeData (const int NodeId);
  inline const int * GetNodeData (const int NodeId) const;

private:

  /// root
  int m_root;
  /// tree data
  FAArray_cont_t < int > m_data;
  /// heap of the deleted nodes
  FAHeap_t < int > m_deleted;
  /// tmp stack
  FAArray_cont_t < int > m_stack;

  /// describes the node
  enum {
    node_type = 0,
    node_offset,
    node_length,
    node_left,
    node_right,
    node_parent,
    node_trbr,
    node_trbr_offset,
    node_size
  };

};

}

#endif
