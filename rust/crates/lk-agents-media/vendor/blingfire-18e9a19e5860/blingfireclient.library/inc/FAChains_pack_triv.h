/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_CHAINS_PACK_TRIV_H_
#define _FA_CHAINS_PACK_TRIV_H_

#include "FAConfig.h"
#include "FASetImageA.h"
#include "FAUtils_cl.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Chains container, encoded by FAChainsPack_triv.
///
/// Note: Also for the given chain the container supports value lookup by its
/// index and EqualOrLess search.
///

class FAChains_pack_triv : public FASetImageA {

public:
    FAChains_pack_triv ();

public:
    /// Sets up image dump
    void SetImage (const unsigned char * pImage);

    /// returns maximum possible size of array UnPack can return
    inline const int GetMaxCount () const;

    /// Decodes the array of values by the specified offset returns its size.
    /// If pValues == NULL the does not use it, stores up to MaxCount values.
    inline const int UnPack (
            const int Offset,
            __out_ecount_opt(MaxCount) int * pValues,
            const int MaxCount
        ) const;

    /// Returns the array of values if it was stored as array of integers
    inline const int UnPack (
            const int Offset,
            const int ** ppValues
        ) const;

    /// Returns value by its index, -1 if the index is out of bounds
    inline const int UnPack (
            const int Offset,
            const int Idx
        ) const;

    /// Finds value or returns closest value less than given if not found
    inline const int GetEqualOrLess (
            const int Offset,
            const int Value,
            int * pIdx = NULL
        ) const;

private:
    // image pointer
    const unsigned char * m_pImage;
    // element size
    int m_SizeOfValue;
    // max number of elements UnPack can return
    int m_MaxCount;

};


inline const int FAChains_pack_triv::GetMaxCount () const
{
    return m_MaxCount;
}


inline const int FAChains_pack_triv::
    UnPack (
        const int Offset,
        __out_ecount_opt(MaxCount) int * pValues,
        const int MaxCount
    ) const
{
    DebugLogAssert (m_pImage);
    DebugLogAssert (sizeof (char) == m_SizeOfValue || \
            sizeof (short) == m_SizeOfValue || \
            sizeof (int) == m_SizeOfValue);

    int Count;

    // copy output weights to the output buffer
    if (sizeof (char) == m_SizeOfValue) {

        // get the number of elements
        Count = *(const char *)(m_pImage + Offset);

        if (NULL != pValues && MaxCount >= Count) {
            // get the pointer to the encoded elements
            const char * pEncodedVals = 
                (const char *)(m_pImage + Offset + sizeof (char));
            // copy elements
            for (int i = 0; i < Count; ++i) {
                const int Val = pEncodedVals [i];
                pValues [i] = Val;
            }
        }
    } else if (sizeof (short) == m_SizeOfValue) {

        // get the number of elements
        Count = *(const short *)(m_pImage + Offset);

        if (NULL != pValues && MaxCount >= Count) {
            // get the pointer to the encoded elements
            const short * pEncodedVals = 
                (const short *)(m_pImage + Offset + sizeof (short));
            // copy elements
            for (int i = 0; i < Count; ++i) {
                const int Val = pEncodedVals [i];
                pValues [i] = Val;
            }
        }
    } else {

        // get the number of elements
        Count = *(const int *)(m_pImage + Offset);

        if (NULL != pValues && MaxCount >= Count) {
            // get the pointer to the encoded elements
            const int * pEncodedVals = 
                (const int *)(m_pImage + Offset + sizeof (int));
            // copy elements
            memcpy (pValues, pEncodedVals, sizeof (int) * Count);
        }
    }

    return Count;
}


inline const int FAChains_pack_triv::
    UnPack (const int Offset, const int ** ppValues) const
{
    DebugLogAssert (m_pImage);
    DebugLogAssert (ppValues);

    if (sizeof (int) != m_SizeOfValue) {
        // explicit interface cannot be used
        return -1;
    }

    // get the number of elements
    const int Count = *(const int *)(m_pImage + Offset);
    // get the pointer to the encoded elements
    const int * pEncodedVals = 
        (const int *)(m_pImage + Offset + sizeof (int));

    *ppValues = pEncodedVals;
    return Count;
}


inline const int FAChains_pack_triv::
    UnPack (
        const int Offset,
        const int Idx
    ) const
{
    DebugLogAssert (m_pImage);
    DebugLogAssert (sizeof (char) == m_SizeOfValue || \
            sizeof (short) == m_SizeOfValue || \
            sizeof (int) == m_SizeOfValue);
    DebugLogAssert (0 <= Idx);

    int Count;

    // copy output weights to the output buffer
    if (sizeof (char) == m_SizeOfValue) {

        // get the number of elements
        Count = *(const char *)(m_pImage + Offset);

        if (Idx < Count) {
            // get the pointer to the encoded elements
            const char * pEncodedVals = 
                (const char *)(m_pImage + Offset + sizeof (char));
            // return the value
            const int Value = pEncodedVals [Idx];
            return Value;
        }
    } else if (sizeof (short) == m_SizeOfValue) {

        // get the number of elements
        Count = *(const short *)(m_pImage + Offset);

        if (Idx < Count) {
            // get the pointer to the encoded elements
            const short * pEncodedVals = 
                (const short *)(m_pImage + Offset + sizeof (short));
            // return the value
            const int Value = pEncodedVals [Idx];
            return Value;
        }
    } else {

        // get the number of elements
        Count = *(const int *)(m_pImage + Offset);

        if (Idx < Count) {
            // get the pointer to the encoded elements
            const int * pEncodedVals = 
                (const int *)(m_pImage + Offset + sizeof (int));
            // return the value
            const int Value = pEncodedVals [Idx];
            return Value;
        }
    }

    return -1;
}


inline const int FAChains_pack_triv::
    GetEqualOrLess (
        const int Offset,
        const int Value,
        int * pIdx
    ) const
{
    DebugLogAssert (m_pImage);
    DebugLogAssert (sizeof (char) == m_SizeOfValue || \
            sizeof (short) == m_SizeOfValue || \
            sizeof (int) == m_SizeOfValue);

    int Count;

    // copy output weights to the output buffer
    if (sizeof (char) == m_SizeOfValue) {

        // get the number of elements
        Count = *(const char *)(m_pImage + Offset);

        // get the pointer to the encoded elements
        const char * pEncodedVals = \
            (const char *)(m_pImage + Offset + sizeof (char));

        // find value index
        int Idx;
        if ((const int)(0xFF >> 1) < Value) {
            DebugLogAssert (Value != *(const char *)&Value);
            Idx = Count - 1;
        } else {
            DebugLogAssert (Value == *(const char *)&Value);
            Idx = FAFindEqualOrLess_log (pEncodedVals, Count, (char) Value);
        }
        *pIdx = Idx;
        if (0 > Idx)
            return -1;

        const int FoundValue = pEncodedVals [Idx];
        return FoundValue;

    } else if (sizeof (short) == m_SizeOfValue) {

        // get the number of elements
        Count = *(const short *)(m_pImage + Offset);

        // get the pointer to the encoded elements
        const short * pEncodedVals = \
            (const short *)(m_pImage + Offset + sizeof (short));

        // find value index
        int Idx;
        if ((const int)(0xFFFF >> 1) < Value) {
            DebugLogAssert (Value != *(const short *)&Value);
            Idx = Count - 1;
        } else {
            DebugLogAssert (Value == *(const short *)&Value);
            Idx = FAFindEqualOrLess_log (pEncodedVals, Count, (short) Value);
        }
        *pIdx = Idx;
        if (0 > Idx)
            return -1;

        const int FoundValue = pEncodedVals [Idx];
        return FoundValue;

    } else {
        // get the number of elements
        Count = *(const int *)(m_pImage + Offset);

        // get the pointer to the encoded elements
        const int * pEncodedVals = \
            (const int *)(m_pImage + Offset + sizeof (int));

        // find value index
        const int Idx = FAFindEqualOrLess_log (pEncodedVals, Count, Value);
        *pIdx = Idx;
        if (0 > Idx)
            return -1;

        const int FoundValue = pEncodedVals [Idx];
        return FoundValue;
    }
}

}

#endif
