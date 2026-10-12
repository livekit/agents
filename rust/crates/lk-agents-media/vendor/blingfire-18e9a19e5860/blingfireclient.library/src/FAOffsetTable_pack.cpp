/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAOffsetTable_pack.h"
#include "FAEncodeUtils.h"

namespace BlingFire
{

FAOffsetTable_pack::FAOffsetTable_pack () :
    m_pBase (NULL),
    m_pDelta (0),
    m_BaseSize (0),
    m_ShiftValue (0),
    m_OffsetCount (0)
{
}

void FAOffsetTable_pack::SetImage (const unsigned char * pImage)
{
    if (pImage) {

        int Offset = 0;

        // get shift-value
        m_ShiftValue = *(const int unsigned *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 <= m_ShiftValue);

        // get base size in bytes
        m_BaseSize = *(const int unsigned *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (sizeof (char) <= (unsigned int) m_BaseSize && \
                  sizeof (int) >= (unsigned int) m_BaseSize);

        // get the number of offset stored in table
        m_OffsetCount = *(const unsigned int *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 < m_OffsetCount);

        // get delta array pointer
        if (0 != m_ShiftValue) {

            m_pDelta = pImage + Offset;
            Offset += m_OffsetCount;

        } else {

            m_pDelta = NULL;
        }

        // get base array pointer
        m_pBase = pImage + Offset;
    }
}

const unsigned int FAOffsetTable_pack::GetOffset (const int Idx) const
{
    DebugLogAssert (0 <= Idx && m_OffsetCount > Idx);

    unsigned int Base;

    // get Base index in m_pBase array
    const int BaseIdx = Idx >> m_ShiftValue;

    FADecode_1_2_3_4_idx (m_pBase, BaseIdx, Base, m_BaseSize);

    // see whether offset table is compressed
    if (NULL != m_pDelta) {

        const unsigned int Delta = m_pDelta [Idx];
        return Base + Delta;

    } else {

        return Base;
    }
}

}
