/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_GETIWS_PACK_TRIV_H_
#define _FA_GETIWS_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAGetIWsCA.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class is able to interpret automaton image stored by FADfaPack_triv
///

class FAGetIWs_pack_triv : public FASetImageA, 
                           public FAGetIWsCA {

public:
    FAGetIWs_pack_triv ();

public:
    /// sets up the image dump and makes the container ready
    void SetImage (const unsigned char * pAutImage);

    /// returns a set of Iws for the given Dfa state
    const int GetIWs (
            const int State,
            __out_ecount_opt (MaxIwCount) int * pIws, 
            const int MaxIwCount
        ) const;

private:
    // pointer to the automaton image
    const unsigned char * m_pAutImage;
    // dst size
    int m_DstSize;
};

}

#endif
