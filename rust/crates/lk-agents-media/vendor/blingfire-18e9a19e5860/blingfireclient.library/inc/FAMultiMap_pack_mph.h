/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAP_PACK_MPH_H_
#define _FA_MULTIMAP_PACK_MPH_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAMultiMapCA.h"
#include "FARSDfa_pack_triv.h"
#include "FAOw2Iw_pack_triv.h"
#include "FAMphInterpretTools_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class interprets dump created by FAMultiMapPack_mph.
///

class FAMultiMap_pack_mph : public FASetImageA,
                            public FAMultiMapCA {
public:
    FAMultiMap_pack_mph ();

public:
    void SetImage (const unsigned char * pDump);

public:
    const int Get (const int Key, const int ** ppValues) const;

    const int Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const;

    const int GetMaxCount () const;

private:
    // maximum possible output size
    int m_MaxChainSize;
    // direction (l2r or r2l) chains are stored in MPH
    int m_Direction;
    // dfa storage
    FARSDfa_pack_triv m_dfa;
    // ow2iw storage
    FAOw2Iw_pack_triv m_ow2iw;
    // MPH interpreter
    FAMphInterpretTools_t < int > m_mph;

};

}

#endif
