/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_ENCODER_PM_DPM_H_
#define _FA_ENCODER_PM_DPM_H_

#include "FAConfig.h"
#include "FAEncoderA.h"

namespace BlingFire
{

/// Does the following:
/// 1. Makes prefix-mask encoding for the first element of chain
/// 2. Makes delta-prefix-mask for all the other elements
///
/// Note:
/// This encoder suits well for non-descending sequences of ints, with
/// the first special element - the size of the sequence, e.g. [5 1 2 3 3 3].
///

class FAEncoder_pm_dpm : public FAEncoderA {

public:

  const int GetMaxBytes () const;
  const int Encode (const int * pChain,
                    const int Size,
                    unsigned char * pOutBuffer) const;

private:

  enum { MAX_BYTES_PER_INT = 9 };

};

}

#endif
