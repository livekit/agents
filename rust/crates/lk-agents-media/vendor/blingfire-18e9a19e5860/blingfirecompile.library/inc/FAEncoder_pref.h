/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ENCODER_PREF_H_
#define _FA_ENCODER_PREF_H_

#include "FAConfig.h"
#include "FAEncoderA.h"

namespace BlingFire
{

/// Does the following:
///  foreach V \in Chain do
///    if V <= 0x7F then
///      *pOut = (unsigned char) V;
///    else if V <= 0x3FFF then
///      ...
///    else if V <= 0x1FFFFF then
///      ...
///    else if V <= 0xFFFFFFF then
///      ...
///    else if V <= 0xFFFFFFFF then
///      ...
///    end

class FAEncoder_pref : public FAEncoderA {

public:

  const int GetMaxBytes () const;
  const int Encode (const int * pChain,
                    const int Size,
                    unsigned char * pOutBuffer) const;

private:

  enum { MAX_BYTES_PER_INT = 5 };

};

}

#endif
