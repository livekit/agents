/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_OW2IW_PACK_TRIV_H_
#define _FA_OW2IW_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAOw2IwCA.h"
#include "FAChains_pack_triv.h"

namespace BlingFire
{

///
/// FAOw2IwCA memory dump based implementation. Can be initialized from
/// the memory dump created with FADfaPack_triv.
///

class FAOw2Iw_pack_triv : public FASetImageA, 
                          public FAOw2IwCA {

public:
    FAOw2Iw_pack_triv ();

public:
    void SetImage (const unsigned char * pImage);
    const int GetDestIwOw (
            const int State, 
            const int Ow1, 
            int * pIw, 
            int * pOw2
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
