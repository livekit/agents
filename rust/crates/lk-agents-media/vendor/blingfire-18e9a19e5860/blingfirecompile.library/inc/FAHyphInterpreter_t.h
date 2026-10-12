/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_HYPHINTERPRETER_T_H_
#define _FA_HYPHINTERPRETER_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FAHyphConfKeeper_packaged.h"
#include "FAW2SConfKeeper.h"
#include "FAWftConfKeeper.h"
#include "FAHyphInterpreter_core_t.h"
#include "FASegmentationTools_bf_t.h"
#include "FASuffixInterpretTools_t.h"
#include "FASecurity.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Hyphenator interpreter, combines:
///
/// W2H -- core patterns
/// W2H_ALT -- alternative core patterns
/// W2H+W2S -- independent segmentation and hyphenation
/// W2S+W2H -- dependent segmentation and segment hyphenation
///

template < class Ty >
class FAHyphInterpreter_t {

public:
    /// creates uninitialized object
    FAHyphInterpreter_t ();

public:
    void SetLDB (const FAMorphLDB_t < Ty > * pLDB);

public:
    /// see FAHyphInterpreter_core_t.h for details
    const int Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize,
            const bool UseAlt = false
        );

private:
    /// helper, makes W2B initialization
    inline const bool InitW2B (const FAMorphLDB_t < Ty > * pLDB);
    /// returns a hyphenation id
    inline const int GetHyphId (const Ty * pSeg, const int SegLen);

private:
    int m_HyphType;
    bool m_fW2B;
    int m_HyphTypeAlt;
    bool m_fW2BAlt;
    FAHyphInterpreter_core_t < Ty > m_w2h;
    const FAHyphInterpreter_core_t < Ty > * m_pW2h;
    FAHyphInterpreter_core_t < Ty > m_w2h_alt;
    const FAHyphInterpreter_core_t < Ty > * m_pW2hAlt;
    FASegmentationTools_bf_t < Ty > m_w2s;
    FASuffixInterpretTools_t < Ty > m_w2b;
};


template < class Ty >
FAHyphInterpreter_t< Ty >::FAHyphInterpreter_t () :
    m_HyphType (FAFsmConst::HYPH_TYPE_CORE),
    m_fW2B (false),
    m_HyphTypeAlt (FAFsmConst::HYPH_TYPE_CORE),
    m_fW2BAlt (false),
    m_pW2h (NULL),
    m_pW2hAlt (NULL)
{}


template < class Ty >
inline const bool FAHyphInterpreter_t< Ty >::
    InitW2B (const FAMorphLDB_t < Ty > * pLDB)
{
    DebugLogAssert (pLDB);

    const FAWftConfKeeper * pConf = pLDB->GetW2BConf ();
    DebugLogAssert (pConf);

    const bool NoTrUse = pConf->GetNoTrUse ();
    const bool UseNfst = pConf->GetUseNfst ();

    FAAssert (!UseNfst, FAMsg::InitializationError);

    if (false == NoTrUse) {
        m_w2b.SetInTr (pLDB->GetInTr ());
        m_w2b.SetOutTr (pLDB->GetOutTr ());
    }
    m_w2b.SetConf (pConf);

    return pConf->GetRsDfa () && pConf->GetState2Ows () && pConf->GetActs ();
}


template < class Ty >
void FAHyphInterpreter_t< Ty >::
    SetLDB (const FAMorphLDB_t < Ty > * pLDB)
{
    m_HyphType = FAFsmConst::HYPH_TYPE_CORE;
    m_fW2B = false;
    m_HyphTypeAlt = FAFsmConst::HYPH_TYPE_CORE;
    m_fW2BAlt = false;
    m_pW2h = NULL;
    m_pW2hAlt = NULL;

    if (!pLDB) {
        return;
    }

    const FAHyphConfKeeper * pHyphConf = pLDB->GetW2HConf ();
    const FAHyphConfKeeper * pHyphAltConf = pLDB->GetW2HAltConf ();
    const FAW2SConfKeeper * pW2SConf = pLDB->GetW2SConf ();
    const FAWftConfKeeper * pW2BConf = pLDB->GetW2BConf ();

    m_w2h.SetConf (pHyphConf);
    m_w2h_alt.SetConf (pHyphAltConf);

    if (pHyphConf) {

        m_pW2h = & m_w2h;

        m_HyphType = pHyphConf->GetHyphType ();

        // initialize w2s, if needed
        if (pW2SConf && FAFsmConst::HYPH_TYPE_CORE != m_HyphType) {
            m_w2s.SetConf (pW2SConf);
        }

        // initialize w2b, if needed
        if (pW2BConf && pHyphConf->GetNormSegs ()) {
            m_fW2B = InitW2B (pLDB);
        }

        // W2S configuration is needed for this type of hyphenation
        FAAssert (FAFsmConst::HYPH_TYPE_CORE == m_HyphType || pW2SConf, \
            FAMsg::InitializationError);
    }

    if (pHyphAltConf) {

        m_pW2hAlt = & m_w2h_alt;

        // main pattern-set must exist
        FAAssert (pHyphConf, FAMsg::InitializationError);

        m_HyphTypeAlt = pHyphAltConf->GetHyphType ();

        // initialize w2s, if needed
        if (pW2SConf && FAFsmConst::HYPH_TYPE_CORE != m_HyphTypeAlt) {
            m_w2s.SetConf (pW2SConf);
        }

        // initialize w2b, if needed
        if (pW2BConf && pHyphAltConf->GetNormSegs ()) {
            m_fW2BAlt = InitW2B (pLDB);
        }

        // W2S configuration is needed for this type of hyphenation
        FAAssert (FAFsmConst::HYPH_TYPE_CORE == m_HyphTypeAlt || pW2SConf, \
            FAMsg::InitializationError);
    }
}


template < class Ty >
inline const int FAHyphInterpreter_t< Ty >::
    GetHyphId (const Ty * pSeg, const int SegLen)
{
    DebugLogAssert (m_fW2B && pSeg && 0 < SegLen);

    Ty Out [FALimits::MaxWordLen + 1];

    const int OutSize = \
        m_w2b.Process (pSeg, SegLen, Out, FALimits::MaxWordLen + 1);

    if (0 < OutSize && FALimits::MaxWordLen + 1 >= OutSize) {

        int OutWordSize = 0;
        for (;OutWordSize < OutSize; ++OutWordSize) {
            if (0 == Out [OutWordSize])
                break;
        }
        // there should be exactly one word concatenated with a 0
        if (OutWordSize != OutSize - 1) {
            return FAFsmConst::HYPH_SIMPLE_HYPH;
        }

        const int LenDiff = OutWordSize - SegLen;
        if (-1 > LenDiff || 1 < LenDiff) {
            return FAFsmConst::HYPH_SIMPLE_HYPH;
        }
        if (0 != memcmp (Out, pSeg, (OutWordSize - 1) * sizeof (Ty))) {
            return FAFsmConst::HYPH_SIMPLE_HYPH;
        }

        // equal length
        if (0 == LenDiff) {

            const int InLast = pSeg [SegLen - 1];
            const int OutLast = Out [SegLen - 1];
            if (OutLast == InLast)
                return FAFsmConst::HYPH_SIMPLE_HYPH;
            else
                return (OutLast << 4) | FAFsmConst::HYPH_CHANGE_BEFORE;

        // output is one character bigger
        } else if (1 == LenDiff) {

            const int OutLast = Out [OutWordSize - 1];
            return (OutLast << 4) | FAFsmConst::HYPH_ADD_BEFORE;

        // output is one character smaller (we under compared one character)
        } else {
            DebugLogAssert (-1 == LenDiff);
            if (pSeg [OutWordSize - 1] == Out [OutWordSize - 1]) {
                return FAFsmConst::HYPH_DELETE_BEFORE;
            }
        }

    } // of if (0 < OutSize && ...

    return FAFsmConst::HYPH_SIMPLE_HYPH;
}


template < class Ty >
const int FAHyphInterpreter_t< Ty >::
    Process (
        const Ty * pIn,
        const int InSize,
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize,
        const bool UseAlt
    )
{
    int Ends [FALimits::MaxWordSize];

    // validate the input
    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || \
        !pOut || MaxOutSize < InSize) {
        return -1;
    }

    const FAHyphInterpreter_core_t < Ty > * pW2H = m_pW2h;

    if (UseAlt) {
        pW2H = m_pW2hAlt;
    }
    if (!pW2H) {
        return -1;
    }

    // a pattern based hyphenation
    if (FAFsmConst::HYPH_TYPE_CORE == m_HyphType) {

        return pW2H->Process (pIn, InSize, pOut, MaxOutSize);

    // a pattern based hyphenation + hyphenation points at segment boundaries
    } else if (FAFsmConst::HYPH_TYPE_W2H_W2S == m_HyphType)  {

        // do a core hyphenation
        const int Count2 = pW2H->Process (pIn, InSize, pOut, MaxOutSize);

        if (-1 == Count2) {
            return -1;
        }

        // make segmenation
        const int EndCount = \
            m_w2s.Process (pIn, InSize, Ends, FALimits::MaxWordSize);

        // add a hyphen after each segment, except for the last one
        if (!m_fW2B) {

            for (int i = 0; i < EndCount - 1; ++i) {

                const int EndPos = Ends [i];
                DebugLogAssert (0 <= EndPos && EndPos < InSize);
                __analysis_assume (0 <= EndPos && EndPos < InSize);

                pOut [EndPos] = FAFsmConst::HYPH_SIMPLE_HYPH;
            }

        } else {

            int BeginPos = 0;

            for (int i = 0; i < EndCount; ++i) {

                const int EndPos = Ends [i];

                DebugLogAssert (0 <= EndPos && EndPos < InSize);
                __analysis_assume (0 <= EndPos && EndPos < InSize);
                DebugLogAssert (BeginPos <= EndPos);
                __analysis_assume (BeginPos <= EndPos);

                const int Len = EndPos - BeginPos + 1;

                // add a hyphenation point between segments
                if (i + 1 < EndCount) {
                    // prefer hyphenator's choise of hyphen
                    if (FAFsmConst::HYPH_NO_HYPH >= pOut [EndPos]) {
                        pOut [EndPos] = GetHyphId (pIn + BeginPos, Len);
                    }
                }

                BeginPos = EndPos + 1;
            }
        }

        return InSize;

    } else {
        DebugLogAssert (FAFsmConst::HYPH_TYPE_W2S_W2H == m_HyphType);

        // this allows not to check for correctness of hyphenation in the loop
        memset (pOut, FAFsmConst::HYPH_NO_HYPH, sizeof (int) * InSize);

        // make segmenation
        const int EndCount = \
            m_w2s.Process (pIn, InSize, Ends, FALimits::MaxWordSize);

        // check whether no splitting was performed
        if (1 >= EndCount) {

            // do a core hyphenation
            pW2H->Process (pIn, InSize, pOut, MaxOutSize);

        } else {

            // hyphenate every segment and add a hyphenation point after it
            int BeginPos = 0;

            for (int i = 0; i < EndCount; ++i) {

                const int EndPos = Ends [i];

                DebugLogAssert (0 <= EndPos && EndPos < InSize);
                __analysis_assume (0 <= EndPos && EndPos < InSize);
                DebugLogAssert (BeginPos <= EndPos);
                __analysis_assume (BeginPos <= EndPos);

                const int Len = EndPos - BeginPos + 1;

                // do a core hyphenation
                pW2H->Process (pIn + BeginPos, Len, pOut + BeginPos, Len);

                // add a hyphenation point between segments
                if (i + 1 < EndCount) {
                    if (!m_fW2B)
                        pOut [EndPos] = FAFsmConst::HYPH_SIMPLE_HYPH;
                    else
                        pOut [EndPos] = GetHyphId (pIn + BeginPos, Len);
                }

                BeginPos = EndPos + 1;
            }

        } // of if (1 < EndCount) ...

        return InSize;

    } // of if (FAFsmConst::HYPH_TYPE_CORE == m_HyphType) ...
}

}

#endif
