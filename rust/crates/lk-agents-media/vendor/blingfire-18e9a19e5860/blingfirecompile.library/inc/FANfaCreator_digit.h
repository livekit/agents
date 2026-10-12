/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_NFACREATOR_DIGIT_H_
#define _FA_NFACREATOR_DIGIT_H_

#include "FAConfig.h"
#include "FANfaCreator_base.h"

namespace BlingFire
{

class FAAllocatorA;


class FANfaCreator_digit : public FANfaCreator_base {

public:

    FANfaCreator_digit (FAAllocatorA * pAlloc);

public:

    void SetTransition (const int FromState,
                        const int ToState,
                        const int LabelOffset,
                        const int LabelLength);
};

}

#endif
