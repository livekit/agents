/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAMultiMap_pack_fixed.h"
#include "FAEncodeUtils.h"

namespace BlingFire
{

FAMultiMap_pack_fixed::FAMultiMap_pack_fixed () :
    m_pData (NULL),
    m_SizeOfValue (0),
    m_SizeOfArr (0),
    m_MaxCount (0),
    m_MinKey (0),
    m_MaxKey (-1)
{}


void FAMultiMap_pack_fixed::SetImage (const unsigned char * pDump)
{
    m_pData = NULL;
    m_SizeOfArr = 0;
    m_SizeOfValue = 0;
    m_MaxCount = 0;
    m_MinKey = 0;
    m_MaxKey = -1;

    if (NULL != pDump) {

        // get value size
        m_SizeOfValue = *(const unsigned int *)(pDump);
        LogAssert (sizeof (char) == m_SizeOfValue || \
            sizeof (short) == m_SizeOfValue || sizeof (int) == m_SizeOfValue);

        // get maximum array size
        m_MaxCount = *(const int *)(pDump + sizeof (int));
        LogAssert (0 < m_MaxCount);

        m_SizeOfArr = (m_MaxCount + 1) * m_SizeOfValue ;
        LogAssert (0 < m_SizeOfArr);

        // get min key value
        m_MinKey = *(const int *)(pDump + (2 * sizeof (int)));
        LogAssert (0 <= m_MinKey);

        // get max key value
        m_MaxKey = *(const int *)(pDump + (3 * sizeof (int)));
        LogAssert (m_MinKey <= m_MaxKey);

        m_pData = pDump + (4 * sizeof (int));
    }
}


const int FAMultiMap_pack_fixed::GetMaxCount () const
{
    return m_MaxCount;
}


const int FAMultiMap_pack_fixed::
    Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const
{
    if (m_MinKey <= Key && Key <= m_MaxKey) {

        int Count;

        const int Idx = Key - m_MinKey;
        const unsigned int Offset = m_SizeOfArr * Idx;
        const unsigned char * pArr = m_pData + Offset;

        // copy output weights to the output buffer
        if (sizeof (char) == m_SizeOfValue) {

            // get the number of elements
            Count = *(const char *)(pArr);
            if (Count > m_MaxCount)
                return -1;

            if (NULL != pValues && MaxCount >= Count) {
                // get the pointer to the encoded elements
                const char * pEncodedVals = 
                    (const char *)(pArr + sizeof (char));
                // copy elements
                for (int i = 0; i < Count; ++i) {
                    const int Val = pEncodedVals [i];
                    pValues [i] = Val;
                }
            }
        } else if (sizeof (short) == m_SizeOfValue) {

            // get the number of elements
            Count = *(const short *)(pArr);
            if (Count > m_MaxCount)
                return -1;

            if (NULL != pValues && MaxCount >= Count) {
                // get the pointer to the encoded elements
                const short * pEncodedVals = 
                    (const short *)(pArr + sizeof (short));
                // copy elements
                for (int i = 0; i < Count; ++i) {
                    const int Val = pEncodedVals [i];
                    pValues [i] = Val;
                }
            }
        } else {

            // get the number of elements
            Count = *(const int *)(pArr);
            if (Count > m_MaxCount)
                return -1;

            if (NULL != pValues && MaxCount >= Count) {
                // get the pointer to the encoded elements
                const int * pEncodedVals = 
                    (const int *)(pArr + sizeof (int));
                // copy elements
                memcpy (pValues, pEncodedVals, sizeof (int) * Count);
            }
        }

        return Count;
    }

    return -1;
}


const int FAMultiMap_pack_fixed::
    Get (const int Key, const int ** ppValues) const
{
    if (m_MinKey <= Key && Key <= m_MaxKey && \
        sizeof (int) == m_SizeOfValue) {

        const int Idx = Key - m_MinKey;
        const unsigned int Offset = m_SizeOfArr * Idx;
        const unsigned char * pArr = m_pData + Offset;

        const int Count = *(const int *)(pArr);

        if (Count > m_MaxCount)
            return -1;

        if (ppValues)
            *ppValues = (const int *)(pArr + sizeof (int));

        return Count;
    }

    return -1;
}

}
