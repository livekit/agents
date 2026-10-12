/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SETIMAGEA_H_
#define _FA_SETIMAGEA_H_

#include "FAConfig.h"

namespace BlingFire
{

///
/// Image pointer set up interface for objects represented as memory dumps.
///

class FASetImageA {

public:
    virtual void SetImage (const unsigned char * pImage) = 0;

};

}

#endif
