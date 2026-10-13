/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_MULTIMAP_PACK_FIXED_H_
#define _FA_MULTIMAP_PACK_FIXED_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAMultiMapCA.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class interprets dump created by FAMultiMapPack_fixed.
///

class FAMultiMap_pack_fixed : public FASetImageA,
                              public FAMultiMapCA {
public:
    FAMultiMap_pack_fixed ();

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
    // points to encoded arrays
    const unsigned char * m_pData;
    // bytes per value
    unsigned int m_SizeOfValue;
    // bytes per array
    unsigned int m_SizeOfArr;
    // maximum array size
    int m_MaxCount;
    // min key value
    int m_MinKey;
    // min key value
    int m_MaxKey;
};

}

#endif
