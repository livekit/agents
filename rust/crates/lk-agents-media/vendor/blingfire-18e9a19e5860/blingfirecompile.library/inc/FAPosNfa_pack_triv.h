/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_POSNFA_PACK_TRIV_H_
#define _FA_POSNFA_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FARSNfaCA.h"
#include "FAOffsetTable_pack.h"
#include "FAChains_pack_triv.h"
#include "FAIwMap_pack.h"

namespace BlingFire
{

///
/// This class is able to interpret automaton image stored by FAPosNfaPack_triv
///

class FAPosNfa_pack_triv : public FASetImageA,
                           public FARSNfaCA {
public:
    FAPosNfa_pack_triv ();

public:
    void SetImage (const unsigned char * pAutImage);

public:
    const int GetInitials (const int ** ppStates) const;
    const bool IsFinal (const int State) const;
    const int GetDest (
            const int State,
            const int Iw,
            int * pDstStates,
            const int MaxCount
        ) const;

private:
    // pointer to the automaton image
    const unsigned char * m_pAutImage;
    // Iw2Iw map
    FAIwMap_pack m_iw2iw;
    // State -> Offset mapping
    FAOffsetTable_pack m_state2offset;
    // destination sets keeper
    FAChains_pack_triv m_dest_sets;
    // initial state count
    int m_InitialCount;
    // pointer to the array of the initial states
    const int * m_pInitials;
};

}

#endif
