/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MEALYDFA_PACK_TRIV_H_
#define _FA_MEALYDFA_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAMealyDfaCA.h"
#include "FAChains_pack_triv.h"

namespace BlingFire
{

///
/// FAMealyDfaCA memory dump based implementation. Can be initialized from
/// the memory dump created with FADfaPack_triv.
///

class FAMealyDfa_pack_triv : public FASetImageA, 
                             public FAMealyDfaCA {

public:
    FAMealyDfa_pack_triv ();

public:
    void SetImage (const unsigned char * pImage);
    const int GetDestOw (
            const int State, 
            const int Iw,
            int * pOw
        ) const;

private:
    // pointer to the image dump
    const unsigned char * m_pAutImage;
    // initial state
    int m_InitialState;
    // Ows keeper
    FAChains_pack_triv m_UnpackOws;
    // dst size
    int m_DstSize;
};

}

#endif
