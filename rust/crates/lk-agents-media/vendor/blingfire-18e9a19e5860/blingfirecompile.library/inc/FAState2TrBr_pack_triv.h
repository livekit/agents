/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_STATE2TRBR_PACK_TRIV_H_
#define _FA_STATE2TRBR_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAMultiMapCA.h"
#include "FAOffsetTable_pack.h"
#include "FAChains_pack_triv.h"

namespace BlingFire
{

///
/// This class implements FAMultiMapCA interface for the beginning and 
/// ending triangular bracket maps by interpreting the dump created by 
/// FAPosNfaPack_triv.
///

class FAState2TrBr_pack_triv : public FASetImageA,
                               public FAMultiMapCA {

public:
    enum {
        MapTypeTrBrBegin = 1,
        MapTypeTrBrEnd = 2,
    };

public:
    FAState2TrBr_pack_triv (const int MapType);

public:
    void SetImage (const unsigned char * pPosNfaImage);

public:
    const int Get (const int State, int * pTrBrs, const int MaxCount) const;
    const int GetMaxCount () const;
    const int Get (const int State, const int ** ppTrBrs) const;

private:
    inline const int GetTrBrOffset (const int State) const;

private:
    // map type
    const int m_MapType;
    // pointer to the automaton image
    const unsigned char * m_pPosNfaImage;
    // global TrBr offset size in bytes
    int m_SizeOfTrBrOffset;
    // State -> Offset mapping
    FAOffsetTable_pack m_state2offset;
    // TrBr values keeper
    FAChains_pack_triv m_trbrs;
};

}

#endif
