/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef _FA_HYPHINTERPRETER_CORE_T_H_
#define _FA_HYPHINTERPRETER_CORE_T_H_

#include "FAConfig.h"
#include "FALimits.h"
#include "FAFsmConst.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAMultiMapCA.h"
#include "FAUtf32Utils.h"
#include "FAHyphConfKeeper_packaged.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Core hyphenator run-time interpreter.
///

template < class Ty >
class FAHyphInterpreter_core_t {

public:
    /// creates uninitialized object
    FAHyphInterpreter_core_t ();

public:
    void SetConf (const FAHyphConfKeeper * pConf);

public:
    /// Returns HyphId chain, of InSize size. Each HyphId of this chain
    /// corresponds to one input symbol and is one of FAFsmConst::HYPH_* .
    /// The method returns -1 if not ready or bad input and does not do any
    /// output if MaxOutSize < InSize.
    const int Process (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const;

private:
    // returns object into initial state
    inline void Clear ();

private:
    // Moore storage
    const FARSDfaCA * m_pDfa;
    const FAState2OwCA * m_pState2Ow;
    const FAMultiMapCA * m_pI2Info;
    int m_LeftAnchor;
    int m_RightAnchor;
    int m_MinPatLen;
    int m_NoHyphLen;
    // flags
    bool m_IgnoreCase;
    bool m_Ready;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    enum {
        DefMinPatLen = 3,
        DefNoHypId = FAFsmConst::HYPH_NO_HYPH,
        DefUnknown = FAFsmConst::HYPH_UNKNOWN,
        DefConflict = FAFsmConst::HYPH_CONFLICT,
        DefDontCare = FAFsmConst::HYPH_DONT_CARE,
    };

};


template < class Ty >
FAHyphInterpreter_core_t< Ty >::FAHyphInterpreter_core_t () :
    m_pDfa (NULL),
    m_pState2Ow (NULL),
    m_pI2Info (NULL),
    m_LeftAnchor (FAFsmConst::IW_L_ANCHOR),
    m_RightAnchor (FAFsmConst::IW_R_ANCHOR),
    m_MinPatLen (DefMinPatLen),
    m_NoHyphLen (0),
    m_IgnoreCase (false),
    m_Ready (false),
    m_pCharMap (NULL)
{}


template < class Ty >
inline void FAHyphInterpreter_core_t< Ty >::Clear ()
{
    m_pDfa  = NULL;
    m_pState2Ow = NULL;
    m_pI2Info = NULL;
    m_LeftAnchor = FAFsmConst::IW_L_ANCHOR;
    m_RightAnchor = FAFsmConst::IW_R_ANCHOR;
    m_MinPatLen = DefMinPatLen;
    m_NoHyphLen = 0;
    m_IgnoreCase = false;
    m_Ready = false;
    m_pCharMap = NULL;
}


template < class Ty >
void FAHyphInterpreter_core_t< Ty >::
    SetConf (const FAHyphConfKeeper * pConf)
{
    FAHyphInterpreter_core_t< Ty >::Clear ();

    if (NULL == pConf) {
        return;
    }

    m_IgnoreCase = pConf->GetIgnoreCase ();
    m_LeftAnchor = pConf->GetLeftAnchor ();
    m_RightAnchor = pConf->GetRightAnchor ();
    m_MinPatLen = pConf->GetMinPatLen ();
    m_NoHyphLen = pConf->GetNoHyphLen ();
    m_pCharMap = pConf->GetCharMap ();

    m_pDfa = pConf->GetRsDfa ();
    m_pState2Ow = pConf->GetState2Ow ();
    m_pI2Info = pConf->GetI2Info ();

    m_Ready = (NULL != m_pDfa) && (NULL != m_pState2Ow) && \
        (NULL != m_pI2Info) && (0 < m_MinPatLen) && (0 <= m_NoHyphLen);
}


template < class Ty >
const int FAHyphInterpreter_core_t< Ty >::
    Process (
        const Ty * pIn,
        const int InSize,
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize
    ) const
{
    Ty pIn2 [FALimits::MaxWordSize + 2];
    int i;

    if (!m_Ready) {
        return -1;
    }
    if (!pIn || !pOut || InSize > FALimits::MaxWordSize || MaxOutSize < InSize) {
        return -1;
    }

    DebugLogAssert (0 < m_MinPatLen && 0 <= m_NoHyphLen);
    DebugLogAssert (m_pDfa && m_pState2Ow && m_pI2Info);

    // +2 for right anchors' output symbols, just to make fewer checks
    const int In2Size = InSize + 2;

    // setup anchors
    pIn2 [0] = (Ty) m_LeftAnchor;
    pIn2 [In2Size - 1] = (Ty) m_RightAnchor;

    // normalize input case
    if (m_IgnoreCase) {
        for (i = 0 ; i < InSize; ++i) {
            const int InSymbol = (int) pIn [i] ;
            pIn2 [i + 1] = (Ty) FAUtf32ToLower (InSymbol) ;
        }
    } else {
        memcpy (pIn2 + 1, pIn, sizeof (Ty) * InSize);
    }
    // normalize characters, only 1 to 1 mapping is allowed
    if (m_pCharMap) {
        int Co;
        for (i = 0; i < InSize; ++i) {
            const Ty Ci = pIn2 [i + 1] ;
            if (1 == m_pCharMap->Get (Ci, &Co, 1)) {
                pIn2 [i + 1] = (Ty) Co;
            }
        }
    }

    // for skipping left anchor's output symbol, without ifs
    int Js = 1;
    int Je;

    // build the output
    for (i = 0; i < InSize; ++i) {
        pOut [i] = DefUnknown;
    }
    for (int From = 0; From < In2Size - (m_MinPatLen - 1); ++From) {

        int State = m_pDfa->GetInitial ();
        int Final = -1;

        for (i = From; i < In2Size; ++i) {

            const int Iw = pIn2 [i];
            State = m_pDfa->GetDest (State, Iw);

            if (-1 == State) {

                break;

            } else if (m_pDfa->IsFinal (State)) {

                Final = State;

                const int PatId = m_pState2Ow->GetOw (Final);
                DebugLogAssert (0 <= PatId);

                const int * pPat;
                const int PatLen = m_pI2Info->Get (PatId, &pPat);
                DebugLogAssert (0 < PatLen && pPat);
                DebugLogAssert (PatLen == i - From + 1);

                Je = PatLen;

                const int OutOfBound = From + PatLen - InSize;
                if (0 < OutOfBound) {
                    Je -= OutOfBound;
                }

                for (int j = Js; j < Je; ++j) {

                    const int Ow = pPat [j];
                    if (DefDontCare == Ow) {
                        continue;
                    }

                    const int Oi = From + j - 1;
                    DebugLogAssert (0 <= Oi && InSize > Oi);
                    const int CurrOw = pOut [Oi];

                    if (DefUnknown == CurrOw) {
                        pOut [Oi] = Ow;
                    } else if (CurrOw != Ow) {
                        pOut [Oi] = DefConflict;
                    }
                }
            }

        } // of for (i = From; ... 

        // no more skipping, as From > 0 from now on
        Js = 0;

    } // of for (int From = 0; ...

    const int NoHyphCount = m_NoHyphLen < InSize ? m_NoHyphLen : InSize ;

    // disable HY points in the beginning and at the end of the word
    for (i = 0; i < NoHyphCount; ++i) {

        pOut [i] = FAFsmConst::HYPH_NO_HYPH;

        const int j = InSize - i - 2;
        if (0 < j) {
            pOut [j] = FAFsmConst::HYPH_NO_HYPH;
        }
    }

    // aut-off the output for the right anchor, even if it's there
    return InSize;
}

}

#endif
