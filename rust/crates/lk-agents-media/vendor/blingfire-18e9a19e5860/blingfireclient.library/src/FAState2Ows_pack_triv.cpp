/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAState2Ows_pack_triv.h"
#include "FAEncodeUtils.h"
#include "FAFsmConst.h"

namespace BlingFire
{

FAState2Ows_pack_triv::FAState2Ows_pack_triv () :
    m_pAutImage (NULL),
    m_DstSize (FAFsmConst::TRIV_PACK_DEF_DST_SIZE)
{}


void FAState2Ows_pack_triv::SetImage (const unsigned char * pAutImage)
{
    m_pAutImage = pAutImage;

    if (m_pAutImage) {

        // get dst size
        m_DstSize = *(const int *)(m_pAutImage);
        if (1 > m_DstSize || 4 < m_DstSize) {
            m_DstSize = FAFsmConst::TRIV_PACK_DEF_DST_SIZE;
        }

        // get Ows address
        const int OwsOffset = *(const int *)(m_pAutImage + sizeof (int));

        if (0 != OwsOffset) {
            const unsigned char * pOwsData = m_pAutImage + OwsOffset;
            m_UnpackOws.SetImage (pOwsData);
        } else {
            m_UnpackOws.SetImage (NULL);
        }
    }
}


const int FAState2Ows_pack_triv::
    GetOwsOffset (const unsigned char * pStatePtr) const
{
    DebugLogAssert (pStatePtr);

    const unsigned char info = *pStatePtr;

    // get output weight size code, 0 if there are no Ow
    const int OwSizeCode = (info & 0x60) >> 5;

    // check whether there is no output weight at this state
    if (0 == OwSizeCode) {
        return -1;
    }

    // skip info
    pStatePtr += sizeof (char);

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
        FADecode_UC_US_UI(pStatePtr, 0, DstCount, IwSize);
        // skip encoded (DstCount - 1) value
        pStatePtr += IwSize;
        // skip two parallel arrays of Iws and Dsts 
        pStatePtr += ((DstCount + 1) * (m_DstSize + IwSize));
        break;
    }

    // Iw-index array
    case FAFsmConst::TRS_IWIA:
    {
        unsigned int IwBase;
        unsigned int IwMax;

        FADecode_UC_US_UI(pStatePtr, 0, IwBase, IwSize);
        pStatePtr += IwSize;
        FADecode_UC_US_UI(pStatePtr, 0, IwMax, IwSize);
        pStatePtr += IwSize;

        DebugLogAssert (IwMax >= IwBase);
        const unsigned int DstCount = IwMax - IwBase + 1;

        // skip Destination Offsets
        pStatePtr += (m_DstSize * DstCount);
        break;
    }

    // ranges
    case FAFsmConst::TRS_RANGE:
    {
        unsigned int RangeCount;
        // decode (RangeCount - 1) value
        FADecode_UC_US_UI(pStatePtr, 0, RangeCount, IwSize);
        // skip encoded (RangeCount - 1) value
        pStatePtr += IwSize;
        // skip FromIws, ToIws and Dsts
        pStatePtr += ((RangeCount + 1) * (m_DstSize + (IwSize * 2)));
        break;
    }

    // implicit transition
    case FAFsmConst::TRS_IMPL:
    {
        // skip input weight
        pStatePtr += IwSize;
        break;
    }

    }; // of switch (TrType) ...

    int OwsOffset;

    // get the output weight
    if (1 == OwSizeCode) {
        OwsOffset = *(const char *)pStatePtr;
    } else if (2 == OwSizeCode) {
        OwsOffset = *(const short *)pStatePtr;
    } else {
        DebugLogAssert (3 == OwSizeCode);
        OwsOffset = *(const int *)pStatePtr;
    }
    
    return OwsOffset;
}


const int FAState2Ows_pack_triv::
    GetOws (const int State, int * pOws, const int MaxCount) const
{
    DebugLogAssert (m_pAutImage);

    int OwsCount;

    // get pointer to the State
    const unsigned char * pCurrPtr = m_pAutImage + State;

    // get OwsOffset
    const int OwsOffset = GetOwsOffset (pCurrPtr);

    // see whether there is no Ows offset stored
    if (-1 == OwsOffset) {
        return -1;
    }

    // decode the array of output weights
    OwsCount = m_UnpackOws.UnPack (OwsOffset, pOws, MaxCount);

    return OwsCount;
}


const int FAState2Ows_pack_triv::GetMaxOwsCount () const
{
    DebugLogAssert (m_pAutImage);

    return m_UnpackOws.GetMaxCount ();
}

}
