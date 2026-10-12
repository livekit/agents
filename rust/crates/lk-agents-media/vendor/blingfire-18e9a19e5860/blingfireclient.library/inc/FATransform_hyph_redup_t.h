/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_HYPH_REDUP_T_H_
#define _FA_TRANSFORM_HYPH_REDUP_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class transforms hyphenated reduplication in the beginning of 
/// the input sequence into a reduplicant followed by a selected delimiter 
/// and the rest of the sequence.
///
/// For example:
///   "aaab-aaabce" -> "aaab\x02ce"
/// Where 0x02 was selected as a delimiter
///
/// Note:
/// 1. If the transformation is not applicable then Process method return -1.
/// 2. if pIn == pOut then transformation is made in-place.
///

template < class Ty >
class FATransform_hyph_redup_t : public FATransformCA_t < Ty > {

public:
    FATransform_hyph_redup_t ();

public:
    /// sets up delimiter value
    void SetDelim (const Ty Delim);

    /// makes transformation
    /// Note: this function has to be *within* the class definition, because otherwise compiler makes an error
    ///   and generates some warnings about "unreferenced local function has been removed".
    const int Process (
            const Ty * pIn,
            const int InCount,
            __out_ecount(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const
    {
        // as __in_range(0, FALimits::MaxWordSize) const int InCount does not work
        __analysis_assume (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);
        DebugLogAssert (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);

        // we are only interested to find hyphen in the first half of an input
        const int HalfCount = (InCount + 1) >> 1;

        int HyphPos = GetHyphPos (0, pIn, HalfCount);

        while (-1 != HyphPos) {

            // see whether hyphenated reduplication exists
            if (true == IsHyphRedup (pIn, InCount, HyphPos)) {
                break;
            }

            HyphPos = GetHyphPos (HyphPos + 1, pIn, HalfCount);
        }

        // see whether hyphen exists
        if (-1 == HyphPos) {
            return -1;
        }

        // see FATransform_hyph_redup_t< Ty >::GetHyphPos implementation
        __analysis_assume (0 < InCount && 0 <= HyphPos && HalfCount > HyphPos);
        DebugLogAssert (0 < InCount && 0 <= HyphPos && HalfCount > HyphPos);

        // calc output size
        const int OutSize = InCount - HyphPos;

        // as InCount > (2 * HyphPos) which is at most InCount - 1
        __analysis_assume (HyphPos < OutSize);
        DebugLogAssert (HyphPos < OutSize);

        if (MaxOutSize >= OutSize) {

            int pos;

            // check whether transformation is not made in-place
            if (pOut != pIn) {
                // copy everything before hyphen
                for (pos = 0; pos < HyphPos; ++pos) {
                    pOut [pos] = pIn [pos];
                }
            } else {
                // start from HyphPos
                pos = HyphPos;
            }

            // put delimiter
            pOut [pos] = m_Delim;
            pos++;

            // skip the hyphen and the right duplicant
            pIn += ((HyphPos << 1) + 1);

            // copy the rest, if any
            for (; pos < OutSize; ++pos) {
                pOut [pos] = *pIn++;
            }
        }

        return OutSize;
    }

private:
    // returns hyphen position, returns -1 if not found
    inline const int GetHyphPos (
            const int FromPos, 
            const Ty * pIn, 
            const int InCount
        ) const;
    // returns true if hyphenated reduplication exist
    inline const bool IsHyphRedup (
            const Ty * pIn, 
            const int InCount,
            const int HyphPos
        ) const;

private:
    // delimiter
    Ty m_Delim;
    // constants
    enum { 
        DefDelim = 2,
        DefHyphen = '-',
    };
};


template < class Ty >
FATransform_hyph_redup_t< Ty >::
    FATransform_hyph_redup_t () :
        m_Delim (DefDelim)
{
}


template < class Ty >
void FATransform_hyph_redup_t< Ty >::
    SetDelim (const Ty Delim)
{
    m_Delim = Delim;
}


template < class Ty >
inline const int FATransform_hyph_redup_t< Ty >::
    GetHyphPos (const int FromPos, const Ty * pIn, const int InCount) const
{
    DebugLogAssert (0 <= InCount);

    for (int i = FromPos; i < InCount; ++i) {

        DebugLogAssert (pIn);

        if (DefHyphen == pIn [i]) {
            return i;
        }
    }

    return -1;
}


template < class Ty >
inline const bool FATransform_hyph_redup_t< Ty >::
#ifndef NDEBUG
    IsHyphRedup (const Ty * pIn, const int InCount, const int HyphPos) const
#else
    IsHyphRedup (const Ty * pIn, const int, const int HyphPos) const
#endif
{
    DebugLogAssert (pIn && 0 < InCount && InCount >= (HyphPos << 1));

    const Ty * pLeft = pIn;
    const Ty * pRight = pIn + HyphPos + 1;

    for (int pos = 0; pos < HyphPos; ++pos) {

        if (pLeft [pos] != pRight [pos]) {
            return false;
        }
    }

    return true;
}

}

#endif
