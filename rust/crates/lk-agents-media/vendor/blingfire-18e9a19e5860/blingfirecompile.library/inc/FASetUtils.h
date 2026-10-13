/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SET_UTILS_H_
#define _FA_SET_UTILS_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

class FAAllocatorA;

/// Description:
///   Algorithms collection for different set operations
///
/// Note:
///   MaxV - maximum value, should not be very large
///

class FASetUtils {

public:
  /// the default number of sets of results is 3
  FASetUtils (FAAllocatorA * pAlloc);
  ~FASetUtils ();

public:
  /// returns the number of sets of results
  const int GetResCount () const;
  /// adjusts the number of sets to operate with
  void SetResCount (const int ResCount);
  // returns object into the initial state
  void Clear ();

  /// calculates all at once: 
  /// Res1 = Set1 - Set2
  /// Res2 = Set2 - Set1
  void Difference (const int * pSet1, const int Size1, const int Res1,
                   const int * pSet2, const int Size2, const int Res2);
  /// calculates intersection:
  /// Res = Set1 & Set2
  void Intersect (const int * pSet1, const int Size1, 
                  const int * pSet2, const int Size2,
                  const int Res);
  /// calculates union:
  /// Res = Set1 | Set2
  void Union (const int * pSet1, const int Size1, 
              const int * pSet2, const int Size2,
              const int Res);
  /// calculates a union between N sets:
  /// Res = Set1 | ... | SetN
  /// Note: UnionN does not require merging arrays to be sets.
  void UnionN (const int ** pSetArray,
               const int * pSizeArray,
               const int Size,
               const int Res);
  /// calculates union: Res = Res | Set
  void SelfUnion (const int * pSet, const int Size, const int Res);
  /// calculates union: Res = Res & Set
  void SelfIntersect (const int * pSet, const int Size, const int Res);

public:
  // returns set_i size and elements
  // if 0 == size then pointer is not defined
  const int GetRes (const int ** ppSet, const int Res) const;
  // returns size of Set[Res]
  const int GetSize (const int Res) const;
  // returns set_i size and elements via this method set_i's elements 
  // can be setup directly. however, call PrepareRes if set properties 
  // are not guaranteed
  const int GetRes (int ** ppSet, const int Res);
  // sets up the result set, Set[Res] can be invalid after this operation
  void SetRes (const int * pSet, const int Size, const int Res);
  // adds one or more element into set, Set[Res] can be invalid after this operation
  void PushBackRes (const int * pSet, const int Size, const int Res);
  // makes Set[Res] to be Size long
  void Resize (const int Size, const int Res);
  // makes Set[Res] a valid Set
  void PrepareRes (const int Res);

private:

  FAArray_cont_t < FAArray_cont_t < int > * > m_results;
  FAAllocatorA * m_pAlloc;

};

}

#endif
