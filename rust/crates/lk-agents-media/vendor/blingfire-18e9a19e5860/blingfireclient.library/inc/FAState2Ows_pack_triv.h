/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2OWS_PACK_TRIV_H_
#define _FA_STATE2OWS_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAState2OwsCA.h"
#include "FAChains_pack_triv.h"

namespace BlingFire
{

///
/// This class is able to interpret automaton image stored by FAAutPack_triv
///

class FAState2Ows_pack_triv : public FASetImageA,
                              public FAState2OwsCA {
public:
    FAState2Ows_pack_triv ();

/// from FASetImageA
public:
    void SetImage (const unsigned char * pAutImage);

/// from FAState2OwsCA
public:
    const int GetOws (const int State, __out_ecount_opt (MaxCount) int * pOws, const int MaxCount) const;
    const int GetMaxOwsCount () const;

private:
    // helper function, returns OwsOffset of the current state
    inline const int GetOwsOffset (const unsigned char * pStatePtr) const;

private:
    // pointer to the automaton image
    const unsigned char * m_pAutImage;
    // decodes Ows sets
    FAChains_pack_triv m_UnpackOws;
    // dst size
    int m_DstSize;
};

}

#endif
