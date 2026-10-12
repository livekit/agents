/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAOw2Iw_pack_triv.h"
#include "FAEncodeUtils.h"
#include "FAFsmConst.h"
#include "FAUtils_cl.h"

namespace BlingFire
{

FAOw2Iw_pack_triv::FAOw2Iw_pack_triv () :
    m_pAutImage (NULL),
    m_InitialState (-1),
    m_DstSize (FAFsmConst::TRIV_PACK_DEF_DST_SIZE)
{}


void FAOw2Iw_pack_triv::SetImage (const unsigned char * pAutImage)
{
    m_pAutImage = pAutImage;

    if (NULL != m_pAutImage) {

        unsigned int Offset = 0;

        // get dst size
        m_DstSize = *(const int *)(m_pAutImage + Offset);
        Offset += sizeof (int);
        if (1 > m_DstSize || 4 < m_DstSize) {
            m_DstSize = FAFsmConst::TRIV_PACK_DEF_DST_SIZE;
        }

        const int OwsOffset = *(const int *)(m_pAutImage + Offset);
        Offset += sizeof (int);
        LogAssert (0 != OwsOffset);

        const unsigned char * pOwsData = m_pAutImage + OwsOffset;
        m_UnpackOws.SetImage (pOwsData);

        const unsigned int * pIwsCount = 
            (const unsigned int *)(m_pAutImage + Offset);
        Offset += sizeof (int);

        // see whether Iws should be remapped
        const bool RemapIws = 0 != (0x80000000 & *pIwsCount);
        /// Iws cannot be remapped for MPH automata
        LogAssert (!RemapIws);

        // get alphabet size
        const int IwCount = 0x7FFFFFFF & *pIwsCount;
        Offset += (sizeof (int) * IwCount);

        // BEWARE, no Iw2Iw map assumed !!!
        m_InitialState = Offset;
    }
}


const int FAOw2Iw_pack_triv::
    GetDestIwOw (
        const int State, 
        const int Ow1, 
        int * pIw, 
        int * pOw2
    ) const
{
    DebugLogAssert (m_pAutImage);
    DebugLogAssert (pIw && pOw2);

    if (0 > State) {
        return -1;
    }

    int DestState = -1;
    int Idx;
    const unsigned char * pOwsOffset = NULL;
    const unsigned char * pCurrPtr = m_pAutImage + State;
    const unsigned char info = *pCurrPtr;

    // skip info
    pCurrPtr++;

    const int IwSize = ((info & 0x18) >> 3) + 1;
    DebugLogAssert (sizeof (char) <= (unsigned int) IwSize && \
            sizeof (int) >= (unsigned int) IwSize);

    // get output weight size code, 0 if there are no Ow
    const int OwSizeCode = (info & 0x60) >> 5;

    const char TrType = info & 0x07;

    switch (TrType) {

    // prallel arrays
    case FAFsmConst::TRS_PARA:
    {
        unsigned int DstCount;

        if (sizeof (char) == IwSize) {
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (char);
            // get OwsOffset pointer: skip Iws and Dst arrays
            pOwsOffset = \
                pCurrPtr + (DstCount * sizeof (char)) + (m_DstSize * DstCount);

        } else if (sizeof (short) == IwSize) {
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned short *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (short);
            // get OwsOffset pointer: skip Iws and Dst arrays
            pOwsOffset = \
                pCurrPtr + (DstCount * sizeof (short)) + (m_DstSize * DstCount);

        } else {
            DebugLogAssert (sizeof (int) == IwSize);
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned int *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (int);
            // get OwsOffset pointer: skip Iws and Dst arrays
            pOwsOffset = \
                pCurrPtr + (DstCount * sizeof (int)) + (m_DstSize * DstCount);
        }

        // check whether there is no output weight at this state
        if (0 != OwSizeCode) {

            int OwsOffset;

            // get the output weight
            if (1 == OwSizeCode) {
                OwsOffset = *(const char *)pOwsOffset;
            } else if (2 == OwSizeCode) {
                OwsOffset = *(const short *)pOwsOffset;
            } else {
                DebugLogAssert (3 == OwSizeCode);
                OwsOffset = *(const int *)pOwsOffset;
            }

            // decode the array of output weights
            *pOw2 = m_UnpackOws.GetEqualOrLess (OwsOffset, Ow1, &Idx);

        } else {

            Idx = 0;
            *pOw2 = 0;
        }

        if (-1 == Idx)
            return -1;

        DebugLogAssert (-1 != *pOw2 && (unsigned int) Idx < DstCount);

        if (sizeof (char) == IwSize) {
            // get the input weight
            const unsigned char * pIws = pCurrPtr;
            *pIw = pIws [Idx];
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (char));

        } else if (sizeof (short) == IwSize) {
            // get the input weight
            const unsigned short * pIws = (const unsigned short *) pCurrPtr;
            *pIw = pIws [Idx];
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (short));

        } else {
            DebugLogAssert (sizeof (int) == IwSize);
            // get the input weight
            const unsigned int * pIws = (const unsigned int *) pCurrPtr;
            *pIw = pIws [Idx];
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (int));
        }

        // position pointer to the destination state
        FADecodeDst_idx (pCurrPtr, Idx, DestState, m_DstSize);

        return DestState;

    } // of case

    // implicit transition
    case FAFsmConst::TRS_IMPL:
    {
        // convert size code into size in bytes
        int OwSize = OwSizeCode;
        if (3 == OwSizeCode) {
            OwSize = sizeof (int);
        }

        // return destination state offset
        if (sizeof (char) == IwSize) {
            DestState = State + sizeof (char) + sizeof (char) + OwSize;
            pOwsOffset = pCurrPtr + sizeof (char);
            *pIw = *pCurrPtr;

        } else if (sizeof (short) == IwSize) {
            DestState = State + sizeof (char) + sizeof (short) + OwSize;
            pOwsOffset = pCurrPtr + sizeof (short);
            *pIw = *(const unsigned short *)pCurrPtr;

        } else {
            DebugLogAssert (sizeof (int) == IwSize);
            DestState = State + sizeof (char) + sizeof (int) + OwSize;
            pOwsOffset = pCurrPtr + sizeof (int);
            *pIw = *(const unsigned int *)pCurrPtr;
        }

        // check whether there is no output weight at this state
        if (0 != OwSizeCode) {

            int OwsOffset;

            // get the output weight
            if (1 == OwSizeCode) {
                OwsOffset = *(const char *)pOwsOffset;
            } else if (2 == OwSizeCode) {
                OwsOffset = *(const short *)pOwsOffset;
            } else {
                DebugLogAssert (3 == OwSizeCode);
                OwsOffset = *(const int *)pOwsOffset;
            }
            // find the output weight
            *pOw2 = m_UnpackOws.GetEqualOrLess (OwsOffset, Ow1, &Idx);

            if (-1 == Idx)
                return -1;

        } else {

            Idx = 0;
            *pOw2 = 0;
        }

        return DestState;

    } // of case

    // Iw-index array and ranges are not currently supported
    case FAFsmConst::TRS_IWIA:
    case FAFsmConst::TRS_RANGE:
    {
        DebugLogAssert (0);
        return -1;
    } // of case

    default:
    {
        DebugLogAssert (FAFsmConst::TRS_NONE == TrType);
        return -1;
    } // of default:

    }; // of switch (TrType)
}

}
