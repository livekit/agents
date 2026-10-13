/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FARSDfa_pack_triv.h"
#include "FAEncodeUtils.h"
#include "FAUtils_cl.h"
#include "FAFsmConst.h"

namespace BlingFire
{

FARSDfa_pack_triv::FARSDfa_pack_triv () :
    m_pAutImage (NULL),
    m_IwCount (0),
    m_pIws (NULL),
    m_InitialState (0),
    m_RemapIws (false),
    m_DstSize (FAFsmConst::TRIV_PACK_DEF_DST_SIZE)
{}


void FARSDfa_pack_triv::SetImage (const unsigned char * pAutImage)
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

        // skip Ows table offset
        Offset += sizeof (int);

        const unsigned int * pIwsCount = 
            (const unsigned int *)(m_pAutImage + Offset);
        Offset += sizeof (int);

        // see whether Iws should be remapped
        m_RemapIws = 0 != (0x80000000 & *pIwsCount);

        // get alphabet size
        m_IwCount = 0x7FFFFFFF & *pIwsCount;

        // get pointer to the alphabet
        m_pIws = (const int *) (m_pAutImage + Offset);
        Offset += (sizeof (int) * m_IwCount);

        LogAssert (m_pIws && 0 < m_IwCount && 0 == m_IwCount % 2);

        // get the initial state
        if (m_RemapIws) {

            // get Iw2Iw dump size
            const int Iw2IwSize = *(const int *)(m_pAutImage + Offset);
            Offset += sizeof (int);

            m_iw2iw.SetImage (m_pAutImage + Offset);
            Offset += Iw2IwSize;
        }

        m_InitialState = Offset;

        LogAssert (FAIsValidDfa (this));
    }
}


const int FARSDfa_pack_triv::GetMaxState () const
{
    return 0xffffff;
}


const int FARSDfa_pack_triv::GetMaxIw () const
{
    DebugLogAssert (m_pIws && 0 < m_IwCount && 0 == m_IwCount % 2);
    return m_pIws [m_IwCount - 1];
}


const int FARSDfa_pack_triv::GetInitial () const
{
    return m_InitialState;
}


const int FARSDfa_pack_triv::
    GetIWs (__out_ecount_opt (MaxIwCount) int * pIws, const int MaxIwCount) const
{
    DebugLogAssert (m_pIws && 0 < m_IwCount && 0 == m_IwCount % 2);

    if (NULL == pIws && 0 != MaxIwCount) {
        return -1;
    }

    int IwCount = 0;

    for (int i = 0; i < m_IwCount; ++i) {

        const int IwFrom = m_pIws [i++];
        const int IwTo = m_pIws [i];

        for (int Iw = IwFrom; Iw <= IwTo; ++Iw) {

            if (0 <= IwCount && IwCount < MaxIwCount) {
                pIws [IwCount] = Iw;
            }

            IwCount++;
        }
    } // for (int i = 0; ...

    return IwCount;
}


const bool FARSDfa_pack_triv::IsFinal (const int State) const
{
    if (0 > State) {
        return false;
    }

    DebugLogAssert (0 < State);

    unsigned char info = m_pAutImage [State];
    return 0 != (0x80 & info);
}


const int FARSDfa_pack_triv::GetDest (const int State, const int Iw) const
{
    if (0 > State) {
        return -1;
    }

    int NewIw;

    if (m_RemapIws) {

        NewIw = m_iw2iw.GetNewIw (Iw);
        if (-1 == NewIw)
            return -1;

    } else {

        NewIw = Iw;
    }

    const unsigned char * pCurrPtr = m_pAutImage + State;
    const unsigned char info = *pCurrPtr;

    // skip info
    pCurrPtr++;

    const int IwSize = ((info & 0x18) >> 3) + 1;
    DebugLogAssert (sizeof (char) <= (unsigned int) IwSize && \
            sizeof (int) >= (unsigned int) IwSize);

    const char TrType = info & 0x07;

    switch (TrType) {

    // prallel arrays
    case FAFsmConst::TRS_PARA:
    {
        int Idx;
        unsigned int DstCount;

        if (sizeof (char) == IwSize) {

            // check whether NewIw is out of bounds for this state
            if (0xFFFFFF00 & NewIw) {
                return -1;
            }
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (char);
            // find outgoing transition index, if any
            const unsigned char * pIws = pCurrPtr;
            Idx = FAFind_log (pIws, DstCount, (unsigned char) NewIw);
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (char));

        } else if (sizeof (short) == IwSize) {

            // check whether NewIw is out of bounds for this state
            if (0xFFFF0000 & NewIw) {
                return -1;
            }
            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned short *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (short);
            // find outgoing transition index, if any
            const unsigned short * pIws = (const unsigned short*) pCurrPtr;
            Idx = FAFind_log (pIws, DstCount, (unsigned short) NewIw);
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
            Idx = FAFind_log (pIws, DstCount, (unsigned int) NewIw);
            // skip Iws array
            pCurrPtr += (DstCount * sizeof (int));
        }

        // transition with NewIw does not exist
        if (-1 == Idx)
            return -1;

        // position pointer to the destination state

        int Dst;
        FADecodeDst_idx (pCurrPtr, Idx, Dst, m_DstSize);
        return Dst;

    } // of case

    // Iw-index array
    case FAFsmConst::TRS_IWIA:
    {
        unsigned int IwBase;
        unsigned int IwMax;

        FADecode_UC_US_UI(pCurrPtr, 0, IwBase, IwSize);
        pCurrPtr += IwSize;
        FADecode_UC_US_UI(pCurrPtr, 0, IwMax, IwSize);
        pCurrPtr += IwSize;

        if (NewIw < (int) IwBase || NewIw > (int) IwMax) {
            return -1;
        }

        const int Idx = NewIw - IwBase;

        // position pointer to the destination state

        int Dst;
        FADecodeDst_idx (pCurrPtr, Idx, Dst, m_DstSize);

        // see whether transition does not exist
        if (0 == Dst) {
            return -1;
        } else {
            return Dst;
        }

    } // of case

    // rangse if Iws
    case FAFsmConst::TRS_RANGE:
    {
        int Idx;
        unsigned int RangeCount;

        if (sizeof (char) == IwSize) {

            // check whether NewIw is out of bounds for this state
            if (0xFFFFFF00 & NewIw) {
                return -1;
            }
            // as RangeCount - 1 was actually encoded
            RangeCount = 1 + *pCurrPtr;
            // skip RangeCount
            pCurrPtr += sizeof (char);
            // find range index
            const unsigned char * pFromIws = pCurrPtr;
            Idx = FAFindEqualOrLess_log \
                (pFromIws, RangeCount, (unsigned char) NewIw);
            // smaller than smallest range beginning
            if (-1 == Idx) {
                return -1;
            }
            // skip FromIws array
            pCurrPtr += (RangeCount * sizeof (char));
            // find range index
            const unsigned char * pToIws = pCurrPtr;
            // the NewIw is outside the range
            if (pToIws [Idx] < (unsigned char) NewIw) {
                return -1;
            }
            // skip ToIws array
            pCurrPtr += (RangeCount * sizeof (char));

        } else if (sizeof (short) == IwSize) {

            // check whether NewIw is out of bounds for this state
            if (0xFFFF0000 & NewIw) {
                return -1;
            }
            // as RangeCount - 1 was actually encoded
            RangeCount = 1 + *(const unsigned short *)pCurrPtr;
            // skip RangeCount
            pCurrPtr += sizeof (short);
            // find range index
            const unsigned short * pFromIws = (const unsigned short*)pCurrPtr;
            Idx = FAFindEqualOrLess_log \
                (pFromIws, RangeCount, (unsigned short) NewIw);
            // smaller than smallest range beginning
            if (-1 == Idx) {
                return -1;
            }
            // skip FromIws array
            pCurrPtr += (RangeCount * sizeof (short));
            // find range index
            const unsigned short * pToIws = (const unsigned short*)pCurrPtr;
            // the NewIw is outside the range
            if (pToIws [Idx] < (unsigned short) NewIw) {
                return -1;
            }
            // skip ToIws array
            pCurrPtr += (RangeCount * sizeof (short));

        } else {
            DebugLogAssert (sizeof (int) == IwSize);

            // as RangeCount - 1 was actually encoded
            RangeCount = 1 + *(const unsigned int *)pCurrPtr;
            // skip RangeCount
            pCurrPtr += sizeof (int);
            // find range index
            const unsigned int * pFromIws = (const unsigned int*)pCurrPtr;
            Idx = FAFindEqualOrLess_log \
                (pFromIws, RangeCount, (unsigned int) NewIw);
            // smaller than smallest range beginning
            if (-1 == Idx) {
                return -1;
            }
            // skip FromIws array
            pCurrPtr += (RangeCount * sizeof (int));
            // find range index
            const unsigned int * pToIws = (const unsigned int*)pCurrPtr;
            // the NewIw is outside the range
            if (pToIws [Idx] < (unsigned int) NewIw) {
                return -1;
            }
            // skip ToIws array
            pCurrPtr += (RangeCount * sizeof (int));
        }

        // position pointer to the destination state

        int Dst;
        FADecodeDst_idx (pCurrPtr, Idx, Dst, m_DstSize);
        return Dst;
    }

    // implicit transition
    case FAFsmConst::TRS_IMPL:
    {
        // get output weight size code, 0 if there are no Ow
        const int OwSizeCode = (info & 0x60) >> 5;

        // convert size code into size in bytes
        int OwSize = OwSizeCode;
        if (3 == OwSizeCode) {
            OwSize = sizeof (int);
        }

        // return destination state offset
        if (sizeof (char) == IwSize) {
            if (NewIw == *pCurrPtr)
                return State + sizeof (char) + sizeof (char) + OwSize;
        } else if (sizeof (short) == IwSize) {
            if (NewIw == *(const unsigned short *)pCurrPtr)
                return State + sizeof (char) + sizeof (short) + OwSize;
        } else {
            DebugLogAssert (sizeof (int) == IwSize);
            if ((unsigned int) NewIw == *(const unsigned int *)pCurrPtr)
                return State + sizeof (char) + sizeof (int) + OwSize;
        }

        return -1;
    } // of case

    }; // of switch (TrType)

    DebugLogAssert (FAFsmConst::TRS_NONE == TrType);
    return -1;
}

}
