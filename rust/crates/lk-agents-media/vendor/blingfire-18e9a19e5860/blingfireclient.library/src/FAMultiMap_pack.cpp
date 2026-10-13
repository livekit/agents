/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAMultiMap_pack.h"
#include "FAEncodeUtils.h"

namespace BlingFire
{

FAMultiMap_pack::FAMultiMap_pack () :
    m_pOffsets (NULL),
    m_MaxKey (0),
    m_SizeOfOffset (0)
{}


void FAMultiMap_pack::SetImage (const unsigned char * pDump)
{
    if (NULL != pDump) {

        unsigned int Offset = 0;

        // get max Key
        m_MaxKey = *(const unsigned int *)(pDump + Offset);
        Offset += sizeof (int);

        // get size of offset
        m_SizeOfOffset = *(const unsigned int *)(pDump + Offset);
        Offset += sizeof (int);

        LogAssert (sizeof (char) <= (unsigned int) m_SizeOfOffset && \
                   sizeof (int) >= (unsigned int) m_SizeOfOffset);

        // get offset array pointer
        m_pOffsets = pDump + Offset;
        Offset += (m_SizeOfOffset * (1 + m_MaxKey));

        // skip mis-aligned bytes
        const int MisAligned = Offset % sizeof (int);
        if (0 != MisAligned) {
            Offset -= MisAligned;
            Offset += sizeof (int);
        }

        // set up m_values' dump
        m_values.SetImage (pDump + Offset);
    }
}


const int FAMultiMap_pack::GetMaxCount () const
{
    const int MaxCount = m_values.GetMaxCount ();
    return MaxCount;
}


inline const unsigned int FAMultiMap_pack::
    GetValsOffset (const int Key) const
{
    DebugLogAssert (m_pOffsets);

    unsigned int ValOffset;

    FADecode_1_2_3_4_idx(m_pOffsets, Key, ValOffset, m_SizeOfOffset);

    return ValOffset;
}


const int FAMultiMap_pack::
    Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const
{
    if (0 <= Key && m_MaxKey >= (unsigned int) Key) {

        // decode values offset
        const unsigned int ValOffset = GetValsOffset (Key);

        if (0 == ValOffset) {
            // no-mapping exists
            return -1;
        }

        // get the values
        const int Count = \
            m_values.UnPack (ValOffset - 1, pValues, MaxCount);

        return Count;

    } else {

        return -1;
    }
}


const int FAMultiMap_pack::Get (const int Key, const int ** ppValues) const
{
    if (0 <= Key && m_MaxKey >= (unsigned int) Key) {

        // decode values offset
        const unsigned int ValOffset = GetValsOffset (Key);

        if (0 == ValOffset) {
            // no-mapping exists
            return -1;
        }

        // get values
        const int Count = m_values.UnPack (ValOffset - 1, ppValues);
        return Count;

    } else {

        return -1;
    }
}

}
