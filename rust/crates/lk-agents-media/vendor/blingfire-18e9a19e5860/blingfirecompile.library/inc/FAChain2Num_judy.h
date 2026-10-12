/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CHAIN2NUM_JUDY_H_
#define _FA_CHAIN2NUM_JUDY_H_

#include "FAConfig.h"

/// #define HAVE_JUDYSL_LIB

#ifndef HAVE_JUDYSL_LIB

#include "FAChain2Num_hash.h"

namespace BlingFire
{

typedef FAChain2Num_hash FAChain2Num_judy;

}

// of ifndef HAVE_JUDYSL_LIB
#else

#include "FAChain2NumA.h"
#include "FAArray_cont_t.h"
#include "FAEncoder_mask.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Notes: if FAEncoderA * is not NULL then uses it before store in the chain
/// otherwise uses the default encoder.
///

class FAChain2Num_judy : public FAChain2NumA {

public:

  FAChain2Num_judy ();
  virtual ~FAChain2Num_judy ();

public:
  // sets up chain encoder
  void SetEncoder (const FAEncoderA * pEncoder);

  const int * Get (const int * pChain, const int Size) const;
  // return value is always 0
  const int Add (const int * pChain, const int Size, const int Value);

  // not supported
  const int GetChainCount () const;
  // not supported
  const int GetChain (const int ChainIdx, const int ** pChain) const;
  // not supported
  const int GetValue (const int Idx) const;

  // removes Chain -> Value pair from the map
  void Remove (const int * pChain, const int Size);

  // returns into the state as if default constructor was called
  void Clear ();
  // sets up allocator
  void SetAllocator (FAAllocatorA * pAlloc);

private:

  enum { MAX_CHAIN_SIZE = 0xfffe };
  enum { CHAIN_BUFF_SIZE = ((MAX_CHAIN_SIZE + 1) * sizeof (int)) };

  void * m_map;
  const FAEncoderA * m_pEncoder;

  unsigned char m_buff [CHAIN_BUFF_SIZE];
  FAEncoder_mask m_def_enc;
};

}

// of ifndef HAVE_JUDYSL_LIB
#endif

// of ifndef _FA_CHAIN2NUM_JUDY_H_
#endif
