/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FALimits.h"
#include "FAArray_pack.h"
#include "FAEncodeUtils.h"

namespace BlingFire
{

FAArray_pack::FAArray_pack () :
    m_M (0),
    m_SizeOfIndex (0),
    m_SizeOfValue (0),
    m_Count (0),
    m_SizeOfChain (0),
    m_pIndex (NULL),
    m_pData (NULL)
{}


void FAArray_pack::SetImage (const unsigned char * pImage)
{
    if (pImage) {

        int Offset = 0;

        m_M = *(const int *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 < m_M && 8 >= m_M);

        m_SizeOfIndex = *(const int *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 <= m_SizeOfIndex && 4 >= m_SizeOfIndex);

        m_SizeOfValue = *(const int *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 < m_SizeOfValue && 4 >= m_SizeOfValue);

        m_Count = *(const int *)(pImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 < m_Count && FALimits::MaxArrSize >= m_Count);

        m_SizeOfChain = m_M * m_SizeOfValue;

        if (0 == m_SizeOfIndex) {

            LogAssert (1 == m_M);
            m_pIndex = NULL;
            m_pData = pImage + Offset;

        } else {

            LogAssert (1 != m_M);
            m_pIndex = pImage + Offset;
            m_pData = pImage + Offset + \
                (int ((m_Count + m_M - 1) / m_M) * m_SizeOfIndex);
        }
    } // of if (pImage) ...
}


const int FAArray_pack::GetAt (const int Idx) const
{
    int Val, ChainIdx;

    DebugLogAssert (0 <= Idx && m_Count > Idx);

    if (1 == m_M) {

        FADecode_1_2_3_4_idx (m_pData, Idx, Val, m_SizeOfValue);

    } else {

        const int Idx1 = Idx / m_M;
        const int Idx2 = Idx % m_M;

        FADecode_1_2_3_4_idx (m_pIndex, Idx1, ChainIdx, m_SizeOfIndex);
        DebugLogAssert (0 <= ChainIdx);

        const unsigned char * pChainDump = \
            m_pData + (ChainIdx * m_SizeOfChain);

        FADecode_1_2_3_4_idx (pChainDump, Idx2, Val, m_SizeOfValue);
    }

    return Val;
}


const int FAArray_pack::GetCount () const
{
    return m_Count;
}

}
