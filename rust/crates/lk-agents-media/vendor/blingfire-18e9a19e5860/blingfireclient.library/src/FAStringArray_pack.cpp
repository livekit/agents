/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FALimits.h"
#include "FAStringArray_pack.h"
#include "FAException.h"

namespace BlingFire
{

FAStringArray_pack::FAStringArray_pack () :
    m_Count (0),
    m_pOffsetsAndEnd (NULL),
    m_pData (NULL)
{}


void FAStringArray_pack::SetImage (const unsigned char * pDump)
{
    if (pDump) {

        // get the number of elements    
        m_Count = *((const unsigned int *) pDump);
        pDump += sizeof (int);
        FAAssert(0 <= m_Count, FAMsg::InternalError);

        if (0 == m_Count) {
            m_pOffsetsAndEnd = NULL;
            m_pData = NULL;
            return;
        }

        // get offsets array to the strings and to the end
        m_pOffsetsAndEnd = ((const unsigned int*)pDump);
        pDump += (sizeof(int) * (m_Count + 1));

        m_pData = ((const unsigned char *)pDump);

        // validate
        for(int i = 0; i < m_Count; ++i) {
            FAAssert(m_pOffsetsAndEnd[i + 1] >= m_pOffsetsAndEnd[i], FAMsg::InternalError);
        }

    } // of if (pDump) ...
}


const int FAStringArray_pack::GetAt (const int Idx, const unsigned char ** ppStr) const
{
    DebugLogAssert(ppStr);

    if (0 > Idx || Idx >= m_Count) {
        return -1;
    }

    const unsigned int Begin = m_pOffsetsAndEnd[Idx];
    const unsigned int End = m_pOffsetsAndEnd[Idx + 1];
    DebugLogAssert(End >= Begin);

    *ppStr = m_pData + Begin;
    return End - Begin;
}


const int FAStringArray_pack::GetAt (
            const int Idx, 
            __out_ecount (MaxBuffSize) unsigned char * pBuff, 
            const int MaxBuffSize
        ) const
{
    if (0 > Idx || Idx >= m_Count) {
        return -1;
    }

    const unsigned int Begin = m_pOffsetsAndEnd[Idx];
    const unsigned int End = m_pOffsetsAndEnd[Idx + 1];
    DebugLogAssert(End >= Begin);

    const int Len = End - Begin;
    for(unsigned int i = 0; i < Len && i < MaxBuffSize; i++) {
        pBuff[i] = m_pData[Begin + i];
    }

    return Len;
}


const int FAStringArray_pack::GetCount () const
{
    return m_Count;
}

}
