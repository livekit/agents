/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_HYPH_REDUP_REV_T_H_
#define _FA_TRANSFORM_HYPH_REDUP_REV_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This class makes reverse to FATransform_hyph_redup_t transformation.
///
/// For example:
///   "aaab\x02ce" -> "aaab-aaabce"
/// Where 0x02 was selected as a delimiter
///
/// Note:
/// 1. If the transformation is not applicable then Process method return -1.
/// 2. if pIn == pOut then transformation is made in-place.
///

template < class Ty >
class FATransform_hyph_redup_rev_t : public FATransformCA_t < Ty > {

public:
    FATransform_hyph_redup_rev_t ();

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

        // find reduplication delimiter
        const int DelimPos = GetDelimPos (pIn, InCount);
        DebugLogAssert (DelimPos < InCount);

        // most of the times it won't be found
        if (-1 == DelimPos) {

            return -1;

        } else {

            // see FATransform_hyph_redup_rev_t< Ty >::GetDelimPos for details
            __analysis_assume (0 <= DelimPos && InCount > DelimPos);
            DebugLogAssert (0 <= DelimPos && InCount > DelimPos);
        
            const int OutSize = InCount + DelimPos;

            if (MaxOutSize >= OutSize) {

                // move or copy suffix, if needed
                const int SuffixLen = InCount - (DelimPos + 1);

                if (0 < SuffixLen) {
                
                    const Ty * pSuffIn = pIn + DelimPos + 1;
                    Ty * pSuffOut = pOut + (DelimPos << 1) + 1;

                    for (int i = SuffixLen - 1; i >= 0; --i) {
                        pSuffOut [i] = pSuffIn [i];
                    }
                }

                // change delimiter to hyphen
                pOut [DelimPos] = DefHyphen;

                // check whether transformation is made in-place
                if (pIn == pOut) {

                    Ty * pPrefOut = pOut + DelimPos + 1;

                    for (int i = 0; i < DelimPos; ++i) {
                        pPrefOut [i] = pIn [i];
                    }

                } else {

                    Ty * pOut2 = pOut + DelimPos + 1;

                    for (int i = 0; i < DelimPos; ++i) {
                        const Ty Symbol = pIn [i];
                        pOut [i] = Symbol;
                        pOut2 [i] = Symbol;
                    }
                }

            } // of if (MaxOutSize >= OutSize) ...

            return OutSize;

        } // of if (-1 == DelimPos) ...
    }

private:
    // returns delimiter position, -1 if not found
    inline const int GetDelimPos (const Ty * pIn, const int InCount) const;

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
FATransform_hyph_redup_rev_t< Ty >::
    FATransform_hyph_redup_rev_t () :
    m_Delim (DefDelim)
{
}


template < class Ty >
void FATransform_hyph_redup_rev_t< Ty >::
    SetDelim (const Ty Delim)
{
    m_Delim = Delim;
}


template < class Ty >
inline const int FATransform_hyph_redup_rev_t< Ty >::
    GetDelimPos (const Ty * pIn, const int InCount) const
{
    for (int i = 0; i < InCount; ++i) {
        if (m_Delim == pIn [i]) {
            return i;
        }
    }

    return -1;
}

}

#endif
