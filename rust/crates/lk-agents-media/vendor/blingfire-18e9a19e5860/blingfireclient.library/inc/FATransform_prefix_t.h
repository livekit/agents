/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_PREFIX_T_H_
#define _FA_TRANSFORM_PREFIX_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FARSDfaCA.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Moves recognized prefix to the end of word delimited with m_Delim value.
///
/// For example: if m_Delim == '_' then "prefrest" -> "rest_pref"
///
/// Notes: 
///   1. Transformation can be made in-place.
///   2. If transformation is made in-place it needs word+pref memory,
///      even if the output result is always word+1 length.
///   3. m_Delim value should not interfere with word's alphabet
///

template < class Ty >
class FATransform_prefix_t : public FATransformCA_t < Ty > {

public:
    FATransform_prefix_t ();

public:
    /// sets up prefix delimiter value
    void SetDelim (const Ty Delim);

    /// sets up an RS automaton of prefixes
    void SetRsDfa (const FARSDfaCA * pPrefDfa);

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
        DebugLogAssert (m_pPrefDfa);

        // as __in_range(0, FALimits::MaxWordSize) const int InCount does not work
        __analysis_assume (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);
        DebugLogAssert (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);


        const int PrefLen = GetPrefLen (pIn, InCount);

        if (0 == PrefLen)
            return -1;

        // see FATransform_prefix_t< Ty >::GetPrefLen implementation
        __analysis_assume (0 < PrefLen && InCount >= PrefLen);
        DebugLogAssert (0 < PrefLen && InCount >= PrefLen);

        // copy
        if (pIn != pOut) {

            // see whether output buffer is large enough
            if (MaxOutSize >= InCount + 1) {
                int i;
                // copy root
                for (i = PrefLen; i < InCount; ++i) {
                    *pOut++ = pIn [i];
                }
                // set-up delimiter
                *pOut++ = m_Delim;
                // copy recognized prefix
                for (i = 0; i < PrefLen; ++i) {
                    *pOut++ = pIn [i];
                }
            }

        // in-place
        } else {

            // see whether output buffer is large enough
            if (MaxOutSize > InCount + PrefLen) {
                int i;
                // save prefix copy
                Ty * pPrefCopy = pOut + InCount;
                for (i = 0; i < PrefLen; ++i) {
                    pPrefCopy [i] = pIn [i];
                }
                // move the root left
                for (i = PrefLen; i < InCount; ++i) {
                    *pOut++ = pIn [i];
                }
                // set-up delimiter
                *pOut++ = m_Delim;
                // move recognized prefix left
                for (i = 0; i < PrefLen; ++i) {
                    *pOut++ = pPrefCopy [i];
                }
            }
        }

        return InCount + 1;
    }

private:
    /// finds the longest known-prefix length
    /// returns 0, if no prefix was recognized
    inline const int GetPrefLen (const Ty * pIn, const int InCount) const;

private:
    // delimiter
    Ty m_Delim;
    // keeps language of prefixes
    const FARSDfaCA * m_pPrefDfa;
    // constants
    enum {
        DefDelim = 3,
    };
};


template < class Ty >
FATransform_prefix_t< Ty >::FATransform_prefix_t () :
    m_Delim (DefDelim),
    m_pPrefDfa (NULL)
{
}


template < class Ty >
void FATransform_prefix_t< Ty >::SetDelim (const Ty Delim)
{
    m_Delim = Delim;
}


template < class Ty >
void FATransform_prefix_t< Ty >::SetRsDfa (const FARSDfaCA * pPrefDfa)
{
    m_pPrefDfa = pPrefDfa;
}


template < class Ty >
inline const int FATransform_prefix_t< Ty >::
    GetPrefLen (const Ty * pIn, const int InCount) const
{
    DebugLogAssert (m_pPrefDfa);

    // as __in_range(0, FALimits::MaxWordSize) const int InCount does not work
    __analysis_assume (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);
    DebugLogAssert (0 < InCount && FALimits::MaxWordSize >= InCount && pIn);

    int PrefLen = 0;
    int State = m_pPrefDfa->GetInitial ();

    // -1 because at least one symbol should be left for root
    for (int i = 0; i < InCount - 1; ++i) {

        DebugLogAssert (pIn);

        // convert input symbol into int
        const int Iw = pIn [i];

        // get go to the following state
        State = m_pPrefDfa->GetDest (State, Iw);

        if (-1 == State) {
            return PrefLen;
        }

        if (m_pPrefDfa->IsFinal (State)) {
            PrefLen = i + 1;
        }
    }

    return PrefLen;
}

}

#endif
