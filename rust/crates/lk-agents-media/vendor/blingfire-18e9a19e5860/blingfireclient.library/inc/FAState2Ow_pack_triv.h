/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OW_PACK_TRIV_H_
#define _FA_STATE2OW_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAState2OwCA.h"

namespace BlingFire
{

///
/// This class is able to interpret automaton image stored by FAAutPack_triv
///

class FAState2Ow_pack_triv : public FASetImageA,
                             public FAState2OwCA {
public:
    FAState2Ow_pack_triv ();

//// from FASetImageA
public:
    void SetImage (const unsigned char * pAutImage);

/// from FAState2OwCA
public:
    const int GetOw (const int State) const;

private:
    // pointer to the automaton image
    const unsigned char * m_pAutImage;
    // dst size
    int m_DstSize;
};

}

#endif
