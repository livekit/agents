/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_TRANSFORM_UNESCAPE_T_H_
#define _FA_TRANSFORM_UNESCAPE_T_H_

#include "FAConfig.h"
#include "FATransformCA_t.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// This transformer first detects if there are escape sequences.
/// If NO such sequences, transformation is NOT applied.
/// If any INVALID escape sequences are found, transformation is NOT applied
///     and the debug version will throw assertion!
/// If all escape sequences are valid, it first replaces all TAB's i.e. \x09's with \x00's.
/// Then it unescapes all escape sequences.
///
/// The valid escape sequences are \t, \r, \n and \\. Anything else are invalid.
///
/// Works as follows:
///
/// "foo\tbar" -> "foo\x09bar"
/// "foo<TAB>bar\tbaz" -> "foo\x00bar\x09baz"
///
/// "foobar" -> -1
/// "foo<TAB>bar" -> -1
///
/// "foo\zbar" -> -1 or debug Assert!
/// "foo\zbar\tbaz" -> -1 or debug Assert!
/// "foo\tbar\zbaz" -> -1 or debug Assert!
///
/// where \ is the escape character
///
/// Note:
/// 1. If NO escape sequences, transformation is NOT applied and Process method returns -1.
/// 2. If any INVALID escape sequences are found or output buffer is too small,
///       transformation is NOT applied and Process method returns -1.
///       The debug version will throw assertion!
/// 3. If all escape sequences are valid, Process method returns count of Ty items after transformation.
/// 4. If pIn == pOut then transformation is made in-place.
/// 5. Language independent but works only with Latin-1 and Unicode.
///

template < class Ty >
class FATransform_unescape_t : public FATransformCA_t < Ty > {

public:
    FATransform_unescape_t ();

public:

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
        __analysis_assume (0 <= InCount && FALimits::MaxWordSize >= InCount && pIn);
        DebugLogAssert (0 <= InCount && FALimits::MaxWordSize >= InCount && pIn);

        const int EscCnt = GetEscSequenceCount (pIn, InCount);

        if (0 == EscCnt) {
            return -1;
        }

        __analysis_assume (0 < EscCnt && InCount >= EscCnt);
        DebugLogAssert (0 < EscCnt && InCount >= EscCnt);

        // check whether the output buffer is large enough.
        if (MaxOutSize < InCount - EscCnt) {
            // output buffer is not large enough, cannot continue.
            // The condition of DebugLogAssert is surely false. Just throw assertion!
            DebugLogAssert (MaxOutSize >= InCount - EscCnt);
            return -1;
        }

        const Ty * pInCur = pIn;
        Ty * pOutCur = pOut;

        for ( int i = 0; i < EscCnt; ++i ) {

            __analysis_assume (InCount >= (pInCur - pIn));
            DebugLogAssert (InCount >= (pInCur - pIn));

            const int InCountCur = InCount - static_cast<int>(pInCur - pIn);
            const Ty * pInEsc = FindEscSequence ( pInCur, InCountCur);
            // Note pInEsc might be == pInCur

            // pInEsc must not be null since we counted number of escape sequences before.
            __analysis_assume (pInEsc && pInEsc >= pInCur);
            DebugLogAssert (pInEsc && pInEsc >= pInCur);

            // copy stuff before the esc seq.
            for ( ; pInCur < pInEsc; ++pOutCur, ++pInCur) {
                *pOutCur = ('\t' == *pInCur) ? 0 : *pInCur;
            }

            //copy the escaped
            pInCur++;
            *pOutCur++ = GetUnescChar(*pInCur);
            pInCur++;
        }

        // copy stuff after the last esc seq
        for ( ; pInCur < pIn + InCount; ++pOutCur, ++pInCur) {
            *pOutCur = ('\t' == *pInCur) ? 0 : *pInCur;
        }

        DebugLogAssert (pOutCur - pOut == InCount - EscCnt);

        return InCount - EscCnt;
    }

private:

    /// The input arg InSym should be the character that immediately follows the escape character.
    /// return the unescaped char if it is an escape sequence.
    /// return -1 if not an escape sequence.
    inline int GetUnescChar (const Ty InSym) const;

    /// If any INVALID escape sequences are found, the debug version will throw assertion!
    /// return 0 if no escape sequences or there are invalid escape sequences.
    /// Otherwise, return the number of valid escape sequences.
    inline int GetEscSequenceCount (const Ty * pIn, const int InCount) const;

    /// return 0/null if no valid escape sequences found.
    /// Otherwise, return the pointer to the 1st valid escape sequence.
    inline const Ty * FindEscSequence (const Ty * pStart, const int Count) const;

private:
    // constants
    enum {
        DefDelim = '\\',
    };
};


template < class Ty >
FATransform_unescape_t< Ty >::
    FATransform_unescape_t ()
{
}



/// The input arg InSym should be the character that immediately follows the escape character.
/// return the unescaped char if it is an escape sequence.
/// return -1 if not an escape sequence.
template < class Ty >
inline int FATransform_unescape_t< Ty >::
    GetUnescChar (const Ty InSym) const
{
    /// TODO: use a map or an array for lookup 
    ///       if a significant # of escape sequences need to be supported.
    switch(InSym)
    {
        case 'n':
            return '\n';

        case 'r':
            return '\r';

        case 't':
            return '\t';
    }

    if (DefDelim == InSym) {
        return DefDelim;
    }
    else {
        return -1;
    }
}


/// If any INVALID escape sequences are found, the debug version will throw assertion!
/// return 0 if no escape sequences or there are invalid escape sequences.
/// Otherwise, return the number of valid escape sequences.
template < class Ty >
inline int FATransform_unescape_t< Ty >::
    GetEscSequenceCount (const Ty * pIn, const int InCount) const
{
    // as __in_range(0, FALimits::MaxWordSize) const int InCount does not work
    __analysis_assume (0 <= InCount && FALimits::MaxWordSize >= InCount && pIn);
    DebugLogAssert (0 <= InCount && FALimits::MaxWordSize >= InCount && pIn);

    int EscCnt = 0;

    // -1 because at least one symbol should be left for escape
    // Note if InCount is 0 or 1, no iteration and we return 0
    for (const Ty * pCur = pIn; pCur < pIn + InCount - 1; ++pCur) {
        if (DefDelim == *pCur) {
            ++pCur;
            if ( -1 == GetUnescChar (*pCur) ) {
                // found an invalid escape sequence.
                // The condition of DebugLogAssert is surely false. Just throw assertion!
                DebugLogAssert (-1 != GetUnescChar (*pCur));
                return 0;
            }
            else {
                EscCnt++;
            }
        }
    }

    return EscCnt;
}


/// return 0/null if no valid escape sequences found.
/// Otherwise, return the pointer to the 1st valid escape sequence.
template < class Ty >
inline const Ty * FATransform_unescape_t< Ty >::
    FindEscSequence (const Ty * pStart, const int Count) const
{
    __analysis_assume (0 <= Count && FALimits::MaxWordSize >= Count && pStart);
    DebugLogAssert (0 <= Count && FALimits::MaxWordSize >= Count && pStart);

    // -1 because at least one symbol should be left for escape
    // Note if Count is 0 or 1, no iteration and we return 0
    for (const Ty * pCur = pStart; pCur < pStart + Count - 1; ++pCur) {
        if (DefDelim == *pCur) {
            ++pCur;
            if ( -1 != GetUnescChar (*pCur) ) {
                return pCur - 1;
            }
        }
    }

    return 0;
}

}

#endif
