/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAP_PACK_H_
#define _FA_MULTIMAP_PACK_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAMultiMapCA.h"
#include "FAChains_pack_triv.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class interprets dump created by FAMultiMapPack.
///

class FAMultiMap_pack : public FASetImageA,
                        public FAMultiMapCA {
public:
    FAMultiMap_pack ();

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
    // returns values offset, assumes a valid Key
    inline const unsigned int GetValsOffset (const int Key) const;

private:
    // dump pointer
    const unsigned char * m_pOffsets;
    // MaxKey
    unsigned int m_MaxKey;
    // Size of offset, e.g. sizeof (Offsets [0])
    int m_SizeOfOffset;
    // values un-packer
    FAChains_pack_triv m_values;
};

}

#endif
