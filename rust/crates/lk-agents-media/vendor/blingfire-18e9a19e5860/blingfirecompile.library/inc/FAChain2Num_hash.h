/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CHAIN2NUM_HASH_H_
#define _FA_CHAIN2NUM_HASH_H_

#include "FAConfig.h"
#include "FAChain2NumA.h"
#include "FAMap_judy.h"
#include "FAArray_t.h"
#include "FAArray_cont_t.h"
#include "FAHeap_t.h"

namespace BlingFire
{

class FAAllocatorA;


class FAChain2Num_hash : public FAChain2NumA {

public:

  FAChain2Num_hash ();
  virtual ~FAChain2Num_hash ();

public:

  /// see FAChain2NumA interface description for details

  void SetEncoder (const FAEncoderA * pEncoder);
  void SetCopyChains (const bool CopyChains);
  const int * Get (const int * pChain, const int Size) const;
  const int Add (const int * pChain, const int Size, const int Value);
  const int GetChainCount () const;
  const int GetChain (const int Idx, const int ** ppChain) const;
  const int GetValue (const int Idx) const;

  /// returns pair's index by the Chain, or -1 if does not exist
  const int GetIdx (const int * pChain, const int Size) const;
  /// sets up allocator
  void SetAllocator (FAAllocatorA * pAlloc);
  // removes Chain -> Value pair from the map
  void Remove (const int * pChain, const int Size);
  /// returns the map in the state as it was just constructed
  void Clear ();

private:
    // generates hash-key for a given chain
    inline static const int Chain2Key (const int * pChain, const int Size);
    // identifies whether chains are equal
    inline static const bool Equal (
            const int * pChain1,
            const int * pChain2,
            const int Size
        );
    // finds index by the given chains or returns -1
    inline const int Chain2Idx (const int * pChain, const int Size) const;
    // associates < chain, value > pair with a new idx, returns new idx
    inline const int AddNewChain (
            const int * pChain, 
            const int Size, 
            const int Value
        );
    // adds new collition set with two initial values
    inline const int AddNewCSet (const int i1, const int i2);

private:
    // Hash key to index mapping:
    // if idx > 0 then 
    //   i = idx - 1;
    // else if idx < 0 then
    //   j = (-idx) - 1;
    // else
    //   DebugLogAssert (0);
    FAMap_judy m_key2idx;
    // Storage of collision sets:
    // j --> { i_0, i_2, ... , i_n }
    FAArray_t < FAArray_cont_t < int > > m_csets;
    // Two parallel arrays (chains and coresponding values):
    // i --> chain, array of the following format: [ N, a_1, a_2, ..., a_N ]
    FAArray_t < int * > m_i2chain;
    // i --> value
    FAArray_t < int > m_i2value;
    // tracks empty deleted i-s due to deleted chains
    FAHeap_t < int > m_i_gaps;
    // tracks empty deleted j-s due to deleted collision sets
    FAHeap_t < int > m_j_gaps;
    // allocator
    FAAllocatorA * m_pAlloc;

};

}

#endif
