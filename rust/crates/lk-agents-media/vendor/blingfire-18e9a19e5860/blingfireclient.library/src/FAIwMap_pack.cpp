/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAIwMap_pack.h"
#include <algorithm>
#include <limits>

namespace BlingFire
{

FAIwMap_pack::FAIwMap_pack () :
    m_IntervalCount (0),
    m_pArrFromIw (NULL),
    m_pArrToIwOffset (NULL),
    m_SizeOfNewIw (0),
    m_pNewIws (NULL),
    m_CacheSize (0),
    m_pIw2IwCache (NULL)
{
}

FAIwMap_pack::~FAIwMap_pack ()
{
    if (m_pIw2IwCache) {
        delete [] m_pIw2IwCache;
        m_pIw2IwCache = NULL;
    }
}

void FAIwMap_pack::SetImage (const unsigned char * pImage)
{
    // reset the cache
    m_CacheSize = 0;
    if (m_pIw2IwCache) {
        delete [] m_pIw2IwCache;
        m_pIw2IwCache = NULL;
    }

    if (pImage) {

        unsigned int Offset = 0;

        // get NewIw + 1 size
        m_SizeOfNewIw = *(const int *)(pImage + Offset);
        Offset += sizeof (int);
        // number of intervals
        m_IntervalCount = *(const int *)(pImage + Offset);
        Offset += sizeof (int);
        // beginnings of the intervals
        m_pArrFromIw = (const int *)(pImage + Offset);
        Offset += (sizeof (int) * m_IntervalCount);
        // array of ToIw, Idx
        m_pArrToIwOffset = (const int *)(pImage + Offset);
        Offset += (2 * sizeof (int) * m_IntervalCount);
        // NewIws storage
        m_pNewIws = pImage + Offset;

        // setup the cache, if the map is not empty
        if (0 < m_IntervalCount) {
            // get pair temporary pointer
            const int * pPair = m_pArrToIwOffset + ((m_IntervalCount - 1) << 1);
            // get ending of the interval
            const int EndIw = *pPair;
            LogAssert (0 <= EndIw);

            // allocate memory for the cache
            const int CacheSize = (std::min) (EndIw + 1, (int)MaxCacheSize);
            m_pIw2IwCache = new int [CacheSize];
            LogAssert (m_pIw2IwCache);

            // initialize the cache
            for (int OldIw = 0; OldIw < CacheSize; ++OldIw)
            {
                // Note: this call will not use cache if m_CacheSize == 0
                const int NewIw = FAIwMap_pack::GetNewIw (OldIw);
                m_pIw2IwCache [OldIw] = NewIw;
            }
            // update cache size
            m_CacheSize = CacheSize;

        } // of if (0 < m_IntervalCount) ...
    }
}

}
