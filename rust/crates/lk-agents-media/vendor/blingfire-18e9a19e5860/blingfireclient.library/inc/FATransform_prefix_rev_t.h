/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_PREFIX_REV_T_H_
#define _FA_TRANSFORM_PREFIX_REV_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Makes reverse transformation to FATransform_prefix_t.
///
/// For example: if m_Delim == '_' then "rest_pref" -> "prefrest"
///
/// Notes: 
///   1. Transformation can be made in-place.
///   2. If transformation is made in-place it needs word+pref memory,
///      even if the output result is always word length.
///   3. m_Delim value should not interfere with word's alphabet
///

template < class Ty >
class FATransform_prefix_rev_t : public FATransformCA_t < Ty > {

public:
    FATransform_prefix_rev_t ();

public:
    /// sets up prefix delimiter value
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


        const int DelimPos = GetDelimPos (pIn, InCount);

        if (-1 == DelimPos)
            return -1;

        // see FATransform_prefix_rev_t< Ty >::GetDelimPos implementation
        __analysis_assume (0 <= DelimPos && InCount > DelimPos);
        DebugLogAssert (0 <= DelimPos && InCount > DelimPos);

        // copy
        if (pIn != pOut) {

            if (MaxOutSize >= InCount - 1) {

                int i;
                // copy prefix
                for (i = DelimPos + 1; i < InCount; ++i) {
                    *pOut++ = pIn [i];
                }
                // copy root
                for (i = 0; i < DelimPos; ++i) {
                    *pOut++ = pIn [i];
                }
            }

        // in-place
        } else {

            const int PrefLen = InCount - DelimPos - 1;

            if (MaxOutSize >= InCount + PrefLen) {

                int i;
                // save prefix copy
                Ty * pPrefCopy = pOut + InCount;
                const Ty * pPref = pIn + DelimPos + 1;
                for (i = 0; i < PrefLen; ++i) {
                    pPrefCopy [i] = pPref [i];
                }
                // move root right
                Ty * pOutEnd = pOut + InCount - 2;
                for (i = DelimPos - 1; i >= 0; --i) {
                    *pOutEnd-- = pIn [i];
                }
                // copy prefix back
                for (i = 0; i < PrefLen; ++i) {
                    *pOut++ = pPrefCopy [i];
                }
            }
        }

        return InCount - 1;
    }

private:
    /// finds delimiter position
    /// returns -1, if transformation cannot be applied
    inline const int GetDelimPos (const Ty * pIn, const int InCount) const;

private:
    // delimiter
    Ty m_Delim;
    // constants
    enum {
        DefDelim = 3,
    };
};


template < class Ty >
FATransform_prefix_rev_t< Ty >::
    FATransform_prefix_rev_t () :
        m_Delim (DefDelim)
{
}


template < class Ty >
void FATransform_prefix_rev_t< Ty >::
    SetDelim (const Ty Delim)
{
    m_Delim = Delim;
}


template < class Ty >
inline const int FATransform_prefix_rev_t< Ty >::
    GetDelimPos (const Ty * pIn, const int InCount) const
{
    DebugLogAssert (pIn && 0 < InCount);

    for (int i = 0; i < InCount; ++i) {
        if (m_Delim == pIn [i])
            return i;
    }

    return -1;
}

}

#endif
