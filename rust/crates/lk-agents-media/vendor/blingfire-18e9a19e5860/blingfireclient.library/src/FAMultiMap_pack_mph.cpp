/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAMultiMap_pack_mph.h"
#include "FAFsmConst.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FAMultiMap_pack_mph::FAMultiMap_pack_mph () :
    m_MaxChainSize (-1),
    m_Direction (FAFsmConst::DIR_L2R)
{}


void FAMultiMap_pack_mph::SetImage (const unsigned char * pDump)
{
    if (pDump) {

        int Offset = 0;

        m_MaxChainSize = *(const int *)(pDump + Offset);
        Offset += sizeof (int);
        LogAssert (0 < m_MaxChainSize);

        m_Direction = *(const int *)(pDump + Offset);
        Offset += sizeof (int);
        LogAssert (FAFsmConst::DIR_L2R == m_Direction || \
                   FAFsmConst::DIR_R2L == m_Direction);

        m_dfa.SetImage (pDump  + Offset);
        m_ow2iw.SetImage (pDump + Offset);
        LogAssert (FAIsValidDfa (&m_dfa));

        m_mph.SetRsDfa (&m_dfa);
        m_mph.SetOw2Iw (&m_ow2iw);
    }
}


const int FAMultiMap_pack_mph::
    Get (const int /*Key*/, const int ** /*ppValues*/) const
{
    // all data are stored implicitly
    return -1;
}


const int FAMultiMap_pack_mph::GetMaxCount () const
{
    return m_MaxChainSize;
}


const int FAMultiMap_pack_mph::
    Get (
            const int Key,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const
{
    const int Count = m_mph.GetChain (Key, pValues, MaxCount);

    // see whether we have to reverse the chain
    if (FAFsmConst::DIR_R2L == m_Direction) {

        // reverse the output chain if it was returned completely
        if (1 < Count && Count <= MaxCount) {

            DebugLogAssert (pValues);
            const int Count_2 = Count >> 1;

            for (int i = 0; i < Count_2; ++i) {

                const int Tmp = pValues [i];
                pValues [i] = pValues [Count - i - 1];
                pValues [Count - i - 1] = Tmp;
            }

        } // of if (Count <= MaxCount) ...

    } // of if (FAFsmConst::DIR_R2L == m_Direction) ...

    return Count;
}

}
