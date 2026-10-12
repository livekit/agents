/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAState2Ow_pack_triv.h"
#include "FAEncodeUtils.h"
#include "FAFsmConst.h"

namespace BlingFire
{

FAState2Ow_pack_triv::FAState2Ow_pack_triv () :
    m_pAutImage (NULL),
    m_DstSize (FAFsmConst::TRIV_PACK_DEF_DST_SIZE)
{}

void FAState2Ow_pack_triv::SetImage (const unsigned char * pAutImage)
{
    m_pAutImage = pAutImage;

    if (NULL != m_pAutImage) {
        // get dst size
        m_DstSize = *(const int *)(m_pAutImage);
        if (1 > m_DstSize || 4 < m_DstSize) {
            m_DstSize = FAFsmConst::TRIV_PACK_DEF_DST_SIZE;
        }
    }
}

const int FAState2Ow_pack_triv::GetOw (const int State) const
{
    DebugLogAssert (m_pAutImage);

    const unsigned char * pCurrPtr = m_pAutImage + State;
    const unsigned char info = *pCurrPtr;

    // get output weight size code, 0 if there are no Ow
    const int OwSizeCode = (info & 0x60) >> 5;

    // check whether there is no output weight at this state
    if (0 == OwSizeCode) {
        return -1;
    }

    // skip info
    pCurrPtr += sizeof (char);

    // calc input weight size
    const int IwSize = ((info & 0x18) >> 3) + 1;
    DebugLogAssert (sizeof (char) <= (unsigned int) IwSize && \
            sizeof (int) >= (unsigned int) IwSize);

    const char TrType = info & 0x07;

    // skip transitions
    switch (TrType) {

    // prallel arrays
    case FAFsmConst::TRS_PARA:
    {
        unsigned int DstCount;
        // decode (DstCount - 1) value
        FADecode_UC_US_UI(pCurrPtr, 0, DstCount, IwSize);
        // skip encoded (DstCount - 1) value
        pCurrPtr += IwSize;
        // skip two parallel arrays of Iws and Dsts 
        pCurrPtr += ((DstCount + 1) * (m_DstSize + IwSize));
        break;
    }

    // Iw-index array
    case FAFsmConst::TRS_IWIA:
    {
        unsigned int IwBase;
        unsigned int IwMax;

        FADecode_UC_US_UI(pCurrPtr, 0, IwBase, IwSize);
        pCurrPtr += IwSize;
        FADecode_UC_US_UI(pCurrPtr, 0, IwMax, IwSize);
        pCurrPtr += IwSize;

        DebugLogAssert (IwMax >= IwBase);
        const unsigned int DstCount = IwMax - IwBase + 1;

        // skip Destination Offsets
        pCurrPtr += (m_DstSize * DstCount);
        break;
    }

    // ranges
    case FAFsmConst::TRS_RANGE:
    {
        unsigned int RangeCount;
        // decode (RangeCount - 1) value
        FADecode_UC_US_UI(pCurrPtr, 0, RangeCount, IwSize);
        // skip encoded (RangeCount - 1) value
        pCurrPtr += IwSize;
        // skip FromIws, ToIws and Dsts
        pCurrPtr += ((RangeCount + 1) * (m_DstSize + (IwSize * 2)));
        break;
    }

    // implicit transition
    case FAFsmConst::TRS_IMPL:
    {
        // skip input weight
        pCurrPtr += IwSize;
        break;
    }

    }; // of switch (TrType) ...

    int Ow;

    // get the output weight
    if (1 == OwSizeCode) {
        Ow = *(const char *)pCurrPtr;
    } else if (2 == OwSizeCode) {
        Ow = *(const short *)pCurrPtr;
    } else {
        DebugLogAssert (3 == OwSizeCode);
        Ow = *(const int *)pCurrPtr;
    }

    return Ow;
}

}
