/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#include "blingfire-client_src_pch.h"
#include "FAConfig.h"
#include "FAGetIWs_pack_triv.h"
#include "FAEncodeUtils.h"
#include "FAFsmConst.h"

namespace BlingFire
{

FAGetIWs_pack_triv::FAGetIWs_pack_triv () :
    m_pAutImage (NULL),
    m_DstSize (FAFsmConst::TRIV_PACK_DEF_DST_SIZE)
{}


void FAGetIWs_pack_triv::SetImage (const unsigned char * pAutImage)
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
        const bool RemapIws = 0 != (0x80000000 & *pIwsCount);
        // this class cannot work with remapped Iws
        LogAssert (!RemapIws);
    }
}



const int FAGetIWs_pack_triv::
    GetIWs (
            const int State,
            __out_ecount_opt (MaxIwCount) int * pOutIws, 
            const int MaxIwCount
        ) const
{
    if (0 > State || !pOutIws) {
        return -1;
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
        unsigned int DstCount;

        if (sizeof (char) == IwSize) {

            // as DstCount - 1 was actually encoded
            DstCount = 1 + *pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (char);
            // copy all outgoing Iws, if any
            const unsigned char * pIws = pCurrPtr;
            if (DstCount <= (unsigned int) MaxIwCount) {
                for (unsigned int i = 0; i < DstCount; ++i) {
                    const int Iw = pIws [i];
                    pOutIws [i] = Iw;
                }
            }
            return DstCount;

        } else if (sizeof (short) == IwSize) {

            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned short *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (short);
            // copy all outgoing Iws, if any
            const unsigned short * pIws = (const unsigned short*) pCurrPtr;
            if (DstCount <= (unsigned int) MaxIwCount) {
                for (unsigned int i = 0; i < DstCount; ++i) {
                    const int Iw = pIws [i];
                    pOutIws [i] = Iw;
                }
            }
            return DstCount;

        } else {
            DebugLogAssert (sizeof (int) == IwSize);

            // as DstCount - 1 was actually encoded
            DstCount = 1 + *(const unsigned int *)pCurrPtr;
            // skip DstCount
            pCurrPtr += sizeof (int);
            // find outgoing transition index, if any
            const unsigned int * pIws = (const unsigned int *) pCurrPtr;
            // copy all outgoing Iws, if any
            if (DstCount <= (unsigned int) MaxIwCount) {
                for (unsigned int i = 0; i < DstCount; ++i) {
                    const int Iw = (int) pIws [i];
                    pOutIws [i] = Iw;
                }
            }
            return DstCount;
        }
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

        int Dst, DstCount = 0;

        for (unsigned int Iw = IwBase; Iw <= IwMax; ++Iw) {

            const int Idx = Iw - IwBase;
            FADecodeDst_idx (pCurrPtr, Idx, Dst, m_DstSize);

            if (0 != Dst) {
                if (DstCount < MaxIwCount) {
                    pOutIws [DstCount] = (int) Iw;
                }
                DstCount++;
            }
        }
        return DstCount;

    } // of case

    // rangse if Iws
    case FAFsmConst::TRS_RANGE:
    {
        unsigned int RangeCount;

        if (sizeof (char) == IwSize) {

            // as RangeCount - 1 was actually encoded
            RangeCount = 1 + *pCurrPtr;
            // skip RangeCount
            pCurrPtr += sizeof (char);
            // find range index
            const unsigned char * pFromIws = pCurrPtr;
            // skip FromIws array
            pCurrPtr += (RangeCount * sizeof (char));
            // find range index
            const unsigned char * pToIws = pCurrPtr;

            int DstCount = 0;

            for (unsigned int i = 0; i < RangeCount; ++i) {

                const int FromIw = (int) pFromIws [i];
                const int ToIw = (int) pToIws [i];

                for (int Iw = FromIw; Iw <= ToIw; ++Iw) {
                    if (DstCount < MaxIwCount) {
                        pOutIws [DstCount] = Iw;
                    }
                    DstCount++;
                }
            }
            return DstCount;

        } else if (sizeof (short) == IwSize) {

            // as RangeCount - 1 was actually encoded
            RangeCount = 1 + *(const unsigned short *)pCurrPtr;
            // skip RangeCount
            pCurrPtr += sizeof (short);
            // find range index
            const unsigned short * pFromIws = (const unsigned short*)pCurrPtr;
            // skip FromIws array
            pCurrPtr += (RangeCount * sizeof (short));
            // find range index
            const unsigned short * pToIws = (const unsigned short*)pCurrPtr;

            int DstCount = 0;

            for (unsigned int i = 0; i < RangeCount; ++i) {

                const int FromIw = (int) pFromIws [i];
                const int ToIw = (int) pToIws [i];

                for (int Iw = FromIw; Iw <= ToIw; ++Iw) {
                    if (DstCount < MaxIwCount) {
                        pOutIws [DstCount] = Iw;
                    }
                    DstCount++;
                }
            }
            return DstCount;

        } else {
            DebugLogAssert (sizeof (int) == IwSize);

            // as RangeCount - 1 was actually encoded
            RangeCount = 1 + *(const unsigned int *)pCurrPtr;
            // skip RangeCount
            pCurrPtr += sizeof (int);
            // find range index
            const unsigned int * pFromIws = (const unsigned int*)pCurrPtr;
            // skip FromIws array
            pCurrPtr += (RangeCount * sizeof (int));
            // find range index
            const unsigned int * pToIws = (const unsigned int*)pCurrPtr;

            int DstCount = 0;

            for (unsigned int i = 0; i < RangeCount; ++i) {

                const int FromIw = (int) pFromIws [i];
                const int ToIw = (int) pToIws [i];

                for (int Iw = FromIw; Iw <= ToIw; ++Iw) {
                    if (DstCount < MaxIwCount) {
                        pOutIws [DstCount] = Iw;
                    }
                    DstCount++;
                }
            }
            return DstCount;
        }
    } // of case

    // implicit transition
    case FAFsmConst::TRS_IMPL:
    {
        if (1 > MaxIwCount) {
            return 1;
        }

        // return destination state offset
        if (sizeof (char) == IwSize) {
            const int Iw = *pCurrPtr;
            *pOutIws = Iw;
        } else if (sizeof (short) == IwSize) {
            const int Iw = *(const unsigned short *)pCurrPtr;
            *pOutIws = Iw;
        } else {
            DebugLogAssert (sizeof (int) == IwSize);
            const int Iw = *(const unsigned int *)pCurrPtr;
            *pOutIws = Iw;
        }

        return 1;

    } // of case

    }; // of switch (TrType)

    DebugLogAssert (FAFsmConst::TRS_NONE == TrType);
    return 0;
}

}
