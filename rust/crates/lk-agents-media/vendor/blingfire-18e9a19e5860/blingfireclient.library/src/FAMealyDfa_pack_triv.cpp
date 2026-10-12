/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAMealyDfa_pack_triv.h"
#include "FAEncodeUtils.h"
#include "FAUtils_cl.h"
#include "FAFsmConst.h"

namespace BlingFire
{

FAMealyDfa_pack_triv::FAMealyDfa_pack_triv () :
    m_pAutImage (NULL),
    m_InitialState (-1),
    m_DstSize (FAFsmConst::TRIV_PACK_DEF_DST_SIZE)
{}


void FAMealyDfa_pack_triv::
    SetImage (const unsigned char * pAutImage)
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

        /// Ows must exist
        LogAssert (0 != OwsOffset);

        const unsigned char * pOwsData = m_pAutImage + OwsOffset;
        m_UnpackOws.SetImage (pOwsData);

        const unsigned int * pIwsCount = 
            (const unsigned int *)(m_pAutImage + Offset);
        Offset += sizeof (int);

        // see whether Iws should be remapped
        const bool RemapIws = 0 != (0x80000000 & *pIwsCount);

        /// Iws cannot be remapped for MealyDfa automata
        LogAssert (!RemapIws);

        // get alphabet size
        const int IwCount = 0x7FFFFFFF & *pIwsCount;
        Offset += (sizeof (int) * IwCount);

        // BEWARE, no Iw2Iw map assumed !!!
        m_InitialState = Offset;
    }
}


const int FAMealyDfa_pack_triv::
    GetDestOw (const int State, const int Iw,  int * pOw) const
{
    DebugLogAssert (m_pAutImage);
    DebugLogAssert (pOw);

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

            // check whether Iw is out of bounds for this state
            if (0xFFFFFF00 & Iw) {
                return -1;
            }
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (char);
            // find outgoing transition index, if any
            const unsigned char * pIws = pCurrPtr;
            Idx = FAFind_log (pIws, DstCount, (unsigned char) Iw);
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (char));

        } else if (sizeof (short) == IwSize) {

            // check whether Iw is out of bounds for this state
            if (0xFFFF0000 & Iw) {
                return -1;
            }
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned short *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (short);
            // find outgoing transition index, if any
            const unsigned short * pIws = (const unsigned short*) pCurrPtr;
            Idx = FAFind_log (pIws, DstCount, (unsigned short) Iw);
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (short));

        } else {
            DebugLogAssert (sizeof (int) == IwSize);

            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned int *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (int);
            // find outgoing transition index, if any
            const unsigned int * pIws = (const unsigned int *) pCurrPtr;
            Idx = FAFind_log (pIws, DstCount, (unsigned int) Iw);
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (int));
        }

        // transition with Iw does not exist
        if (-1 == Idx)
            return -1;

        // check whether there is no output weight at this state
        if (0 != OwSizeCode) {
            // skip Destination Offsets
            pOwsOffset = pCurrPtr + (m_DstSize * DstCount);
        }

        // position pointer to the destination state
        FADecodeDst_idx (pCurrPtr, Idx, DestState, m_DstSize);

        break;
    } // of case

    // implicit transition
    case FAFsmConst::TRS_IMPL:
    {
        Idx = 0;

        // convert size code into size in bytes
        int OwSize = OwSizeCode;
        if (3 == OwSizeCode) {
            OwSize = sizeof (int);
        }

        // return destination state offset
        if (sizeof (char) == IwSize) {
            if (Iw == *pCurrPtr) {
                pOwsOffset = pCurrPtr + sizeof (char);
                DestState = State + sizeof (char) + sizeof (char) + OwSize;
            } else {
                return -1;
            }
        } else if (sizeof (short) == IwSize) {
            if (Iw == *(const unsigned short *)pCurrPtr) {
                pOwsOffset = pCurrPtr + sizeof (short);
                DestState = State + sizeof (char) + sizeof (short) + OwSize;
            } else {
                return -1;
            }
        } else {
            DebugLogAssert (sizeof (int) == IwSize);
            if ((unsigned int) Iw == *(const unsigned int *)pCurrPtr) {
                pOwsOffset = pCurrPtr + sizeof (int);
                DestState = State + sizeof (char) + sizeof (int) + OwSize;
            } else {
                return -1;
            }
        }
        break;
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

    if (0 < OwSizeCode && pOwsOffset) {

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
        const int Ow = m_UnpackOws.UnPack (OwsOffset, Idx);
        *pOw = Ow;

    } else {

        *pOw = -1;
    }

    return DestState;
}

}
