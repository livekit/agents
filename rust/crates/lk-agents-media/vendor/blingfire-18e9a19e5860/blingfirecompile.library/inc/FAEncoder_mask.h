/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ENCODER_MASK_H_
#define _FA_ENCODER_MASK_H_

#include "FAConfig.h"
#include "FAEncoderA.h"

namespace BlingFire
{

/// Does the following:
/// 1. Recodes zero-bytes with two non-zero byte sequences

class FAEncoder_mask : public FAEncoderA {

public:

  const int GetMaxBytes () const;
  const int Encode (const int * pChain,
                    const int Size,
                    unsigned char * pOutBuffer) const;

private:

  enum { MAX_BYTES_PER_INT = sizeof (int) * 2 };

};

}

#endif
