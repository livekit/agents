/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTREE2FUNCS_H_
#define _FA_REGEXPTREE2FUNCS_H_

#include "FAConfig.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;
class FARegexpTree;

///
/// This class calculates five functions needed to construct Nfa by Regexp:
///
/// 1. Nullable: { NodeId -> {true, false}}
///    indicates whether subexpression can match empty chain
/// 2. FirstPos: { NodeId -> {pos1, pos2, ...}}
///    maps subexpression to positions where all possible matches can start
/// 3. LastPos: { NodeId -> {pos1, pos2, ...}}
///    maps subexpression to positions where all possible matches can finish
/// 4. FollowPos: { pos -> {pos1, pos2, ...}}
///    maps Regexp's position to the set of positions which can follow
/// 5. Pos2Node: { pos -> NodeId }
///    maps symbol positions to tree nodes
///

class FARegexpTree2Funcs {

 public:

  FARegexpTree2Funcs (FAAllocatorA * pAlloc);
  ~FARegexpTree2Funcs ();

 public:

  /// sets up input tree
  void SetRegexpTree (const FARegexpTree * pRegexpTree);
  /// makes calculation
  void Process ();

  /// if SetSize == 0 then ppSet pointer is not defined

  /// returns true if the NodeId is nullable
  const bool GetNullable (const int NodeId) const;
  /// returns the size of FirstSet
  const int GetFirstPos (const int NodeId, const int ** ppSet) const;
  /// returns the size of LastSet
  const int GetLastPos (const int NodeId, const int ** ppSet) const;
  /// returns the size of FollowSet
  const int GetFollowPos (const int Pos, const int ** ppSet) const;
  /// returns max pos (min pos == 0)
  const int GetMaxPos () const;
  /// returns NodeId
  const int GetNodeId (const int Pos) const;

 private:

  FAAllocatorA * m_pAlloc;

  typedef FAArray_cont_t < int > _TSet;
  typedef FAArray_t < _TSet * > _TNode2Set;

  // regexp tree
  const FARegexpTree * m_pRegexpTree;
  // functions containers
  FAArray_t < int > m_Node2Pos;
  FAArray_t < int > m_Pos2Node;
  FAArray_t < bool > m_Nullable;

  int m_max_pos;

  _TNode2Set m_FirstPos;
  _TNode2Set m_LastPos;
  _TNode2Set m_FollowPos;

  /// NodeId(s) in topological order
  FAArray_t < int > m_sorted_nodes;
  FAArray_cont_t < int > m_tmp_array;

 private:

  /// initializes containers of the functions
  void Init (const int NodeCount);
  /// returns processor into the initial state
  void Clear ();
  /// sorts tree nodes in the reverse topological order
  void SortNodes ();

  /// builds node -> pos map
  void CalcNode2Pos ();
  /// calculates m_pNullable
  void CalcNullable (const int NodeId);
  /// m_pFirstPos
  void CalcFirst (const int NodeId);
  /// m_pLastPos
  void CalcLast (const int NodeId);
  /// m_FollowPos
  void CalcFollow (const int NodeId);

  /// merges two sets and stores the result into m_tmp_array
  inline static void MergeSets (
      _TSet * pDst,
      const _TSet * pSrc1,
      const _TSet * pSrc2
    );
  /// appends set
  inline static void AppendSet (_TSet * pDst, const _TSet * pSet);
  /// returns set (existing or creates a new one)
  inline _TSet * GetSet (_TNode2Set * pArray, const int Idx);
  /// initializes set
  inline void SetSet (_TNode2Set * pArray, const int Idx, const _TSet * pSet);

};

}

#endif
