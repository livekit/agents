/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_IWMAP_PACK_H_
#define _FA_IWMAP_PACK_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAEncodeUtils.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

///
/// This class interprets dump created by FAIwMapPack
///

class FAIwMap_pack : public FASetImageA {

public:
    FAIwMap_pack ();
    ~FAIwMap_pack ();

public:
    /// sets up image dump
    void SetImage (const unsigned char * pImage);
    /// returns NewIw for the given OldIw
    inline const int GetNewIw (const int OldIw) const;

private:
    // interval count
    int m_IntervalCount;
    // the array of the beginnings of Iw intervals
    const int * m_pArrFromIw;
    // the array of pairs: ToIw, Idx
    const int * m_pArrToIwOffset;
    // the number of bytes necessary to represent (MaxNewIw + 1)
    int m_SizeOfNewIw;
    // the storage array of encoded NewIws
    const unsigned char * m_pNewIws;

    // constants
    enum {
        MaxCacheSize = 0xFFFF,
    };
    int m_CacheSize;
    int * m_pIw2IwCache;
};


inline const int FAIwMap_pack::GetNewIw (const int OldIw) const
{
    DebugLogAssert (1 <= m_SizeOfNewIw && 4 >= m_SizeOfNewIw);
    DebugLogAssert (0 < m_IntervalCount);
    DebugLogAssert (m_pArrFromIw && m_pArrToIwOffset && m_pNewIws);

    // get the Iw from the cache
    if (0 <= OldIw && OldIw < m_CacheSize) {
        DebugLogAssert (m_pIw2IwCache);
        return m_pIw2IwCache [OldIw];
    }

    // decode the Iw
    unsigned int NewIw;

    const int IntervalIdx = \
        FAFindEqualOrLess_log (m_pArrFromIw, m_IntervalCount, OldIw);

    // smaller than smallest interval beginning
    if (-1 == IntervalIdx) {
        return -1;
    }

    // get beginning of the interval
    const int FromIw = m_pArrFromIw [IntervalIdx];
    DebugLogAssert (FromIw <= OldIw);

    // get pair temporary pointer
    const int * pPair = m_pArrToIwOffset + (IntervalIdx << 1);
    // get ending of the interval
    const int EndIw = *pPair;
    pPair++;
    // get encoded NewIws' offset for this interval
    const int IntervalOffset = *pPair;
    DebugLogAssert (0 <= IntervalOffset);

    // bigger than the biggest value in this interval
    if (OldIw > EndIw) {
        return -1;
    }

    // get pointer to the interval of NewIws
    const unsigned char * pNewIws = m_pNewIws + IntervalOffset;

    const int Idx = OldIw - FromIw;
    DebugLogAssert (0 <= Idx);

    FADecode_1_2_3_4_idx (pNewIws, (unsigned int) Idx, NewIw, m_SizeOfNewIw);

    // NewIw == 0, if there is no mapping othewise NewIw + 1 was stored
    if (0 != NewIw)
        return NewIw - 1;
    else
        return -1;
}

}

#endif
