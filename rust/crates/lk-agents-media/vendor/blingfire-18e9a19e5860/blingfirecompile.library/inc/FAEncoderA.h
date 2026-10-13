/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ENCODERA_H_
#define _FA_ENCODERA_H_

#include "FAConfig.h"

namespace BlingFire
{

class FAEncoderA {

public:

  /// returns the maximum possible number of bytes needed to encode one int
  virtual const int GetMaxBytes () const = 0;
  /// makes the encoding, returns the size of the encoded sequence
  virtual const int Encode (const int * pChain,
                            const int Size,
                            unsigned char * pOutBuffer) const = 0;
};

}

#endif
