/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_RSDFA_PACK_TRIV_H_
#define _FA_RSDFA_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FARSDfaCA.h"
#include "FAIwMap_pack.h"

namespace BlingFire
{

///
/// This class is able to interpret automaton image stored by FADfaPack_triv
///

class FARSDfa_pack_triv : public FASetImageA,
                          public FARSDfaCA {
public:
    FARSDfa_pack_triv ();

public:
    void SetImage (const unsigned char * pAutImage);

/// read interface
public:
    const int GetMaxState () const;
    const int GetMaxIw () const;
    const int GetInitial () const;
    const int GetIWs (
            __out_ecount_opt (MaxIwCount) int * pIws, 
            const int MaxIwCount
        ) const;
    const bool IsFinal (const int State) const;
    const int GetDest (const int State, const int Iw) const;

private:
    // interprets iw2iw map dump, if any
    FAIwMap_pack m_iw2iw;
    // pointer to the automaton image
    const unsigned char * m_pAutImage;
    // alphabet size
    int m_IwCount;
    // alphabet pointer
    const int * m_pIws;
    // initial state
    int m_InitialState;
    // indicates whether it is necessary to do input weights remapping
    bool m_RemapIws;
    // dst size
    int m_DstSize;
};

}

#endif
