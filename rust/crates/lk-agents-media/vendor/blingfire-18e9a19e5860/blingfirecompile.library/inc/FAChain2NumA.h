/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CHAIN2NUMA_H_
#define _FA_CHAIN2NUMA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAEncoderA;


class FAChain2NumA {

public:
  // sets up chain encoder
  virtual void SetEncoder (const FAEncoderA * pEncoder) = 0;

  /// finds a Value by a Chain, returns NULL if not found
  virtual const int * Get (const int * pChain, const int Size) const = 0;
  /// adds a Chain -> Value pair, returns ChainIdx using by the GetChain method
  virtual const int Add (const int * pChain,
                         const int Size,
                         const int Value) = 0;
  /// returns the total number of chains stored in the map
  virtual const int GetChainCount () const = 0;
  /// returns Chain by its idx
  ///   idx \in [0, GetChainCount ())
  ///   return value is the size of the Chain
  virtual const int GetChain (const int Idx,
                              const int ** pChain) const = 0;
  /// returns value by its idx
  virtual const int GetValue (const int Idx) const = 0;

};

}

#endif
