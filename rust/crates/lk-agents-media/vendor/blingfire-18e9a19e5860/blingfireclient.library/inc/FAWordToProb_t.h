/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WORDTOPROB_T_H_
#define _FA_WORDTOPROB_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwCA.h"
#include "FAUtils_cl.h"
#include "FAFsmConst.h"
#include "FATransformCA_t.h"
#include "FAW2PConfKeeper.h"
#include "FALimits.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Word --> Probability Guesser
///
/// Notes:
///
/// 1. The Ty is the character type, can be any basic type enough to hold 
///    expected Unicode values.
///

template < class Ty >
class FAWordToProb_t {

public:
    FAWordToProb_t ();
    virtual ~FAWordToProb_t ();

public:
    /// Note: this method expects initialized configuration object
    /// pInTr can be NULL if no transformation assumed
    virtual void SetConf (
            const FAW2PConfKeeper * pConf, 
            const FATransformCA_t < Ty > * pInTr
        );

    /// returns a log prob of the input word
    const float GetProb (
            const Ty * pWordStr, // input word
            const int WordLen    // inpur word length
        );

    /// returns an int value proportional to the log prob of the input word
    const int GetIntProb (
            const Ty * pWordStr, // input word
            const int WordLen    // inpur word length
        );

private:
    /// returns object into the initial state
    inline void Clear ();

    // takes results from the m_pIn and stores into pOut
    inline void ApplyInTr (
            __out_ecount(MaxOutSize) Ty * pOut, 
            const int MaxOutSize
        );

private:
    const FARSDfaCA * m_pDfa;
    const FAState2OwCA * m_pState2Ow;
    const FATransformCA_t < Ty > * m_pInTr;

    const Ty * m_pIn;
    int m_InSize;

    bool m_IgnoreCase;

    int m_MaxIntProb;
    float * m_pInt2Float;

    float m_MinProbVal;
    float m_MaxProbVal;

    bool m_fReady;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    enum {
        DefUnfoundProb = 2 * FAFsmConst::MIN_LOG_PROB,
        DefDelimIw = FAFsmConst::IW_ANY,
    };
};


template < class Ty >
FAWordToProb_t< Ty >::FAWordToProb_t () :
    m_pDfa (NULL),
    m_pState2Ow (NULL),
    m_pInTr (NULL),
    m_pIn (NULL),
    m_InSize (0),
    m_IgnoreCase (false),
    m_MaxIntProb (0),
    m_pInt2Float (NULL),
    m_MinProbVal (0),
    m_MaxProbVal (1),
    m_fReady (false),
    m_pCharMap (NULL)
{
}


template < class Ty >
FAWordToProb_t< Ty >::~FAWordToProb_t ()
{
    FAWordToProb_t< Ty >::Clear ();
}


template < class Ty >
inline void FAWordToProb_t< Ty >::Clear ()
{
    m_IgnoreCase = false;
    m_fReady  = false;
    m_pCharMap = NULL;
    m_MaxIntProb = 0;
    m_MinProbVal = 0;
    m_MaxProbVal = 1;

    if (m_pInt2Float) {
        delete [] m_pInt2Float;
        m_pInt2Float = NULL;
    }
}


template < class Ty >
void FAWordToProb_t< Ty >::SetConf (
        const FAW2PConfKeeper * pConf, 
        const FATransformCA_t < Ty > * pInTr
    )
{
    FAWordToProb_t< Ty >::Clear ();

    DebugLogAssert (false == m_fReady);

    if (NULL == pConf) {
        return;
    }

    m_pDfa = pConf->GetRsDfa ();
    m_pState2Ow = pConf->GetState2Ow ();
    m_IgnoreCase = pConf->GetIgnoreCase ();
    m_pCharMap = pConf->GetCharMap ();

    m_pInTr = pInTr;

    m_MaxIntProb = pConf->GetMaxProb ();

    if (0 < m_MaxIntProb) {

        // allocate the dictionary
        m_pInt2Float = NEW float [m_MaxIntProb + 1];
        LogAssert (m_pInt2Float);

        const float Min = pConf->GetMinProbVal ();
        const float Max = pConf->GetMaxProbVal ();

        for (int i = 0; i <= m_MaxIntProb; ++i) {
            const float LogProb = 
                ((float(i) / float(m_MaxIntProb)) * (Max - Min)) + Min;
            m_pInt2Float [i] = LogProb;
        }
    }

    m_fReady = m_pDfa && m_pState2Ow && m_pInt2Float && 0 < m_MaxIntProb;
}


template < class Ty >
inline void FAWordToProb_t< Ty >::
    ApplyInTr (__out_ecount(MaxOutSize) Ty * pOut, const int MaxOutSize)
{
    DebugLogAssert (m_pInTr);
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    DebugLogAssert (pOut && 0 < MaxOutSize);

    // apply the transformation, possibly in-place
    const int OutSize = m_pInTr->Process (m_pIn, m_InSize, pOut, MaxOutSize);

    // see whether there was some transformation made
    if (0 < OutSize && OutSize <= MaxOutSize) {
        m_pIn = pOut;
        m_InSize = OutSize;
    }
}


template < class Ty >
const int FAWordToProb_t< Ty >::GetIntProb (const Ty * pWordStr, const int WordSize)
{
    DebugLogAssert (pWordStr && 0 < WordSize && FALimits::MaxWordLen >= WordSize);
    __analysis_assume (pWordStr && 0 < WordSize && FALimits::MaxWordLen >= WordSize);

    if (!m_fReady) {
        return -1;
    }

    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];

    m_pIn = pWordStr;
    m_InSize = WordSize;

    // normalize case, if needed
    if (m_IgnoreCase) {
        for (int i = 0 ; i < WordSize; ++i) {
            const int InSymbol = (int) m_pIn [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            TmpBuff [i] = (Ty) OutSymbol ;
        }
        m_pIn = TmpBuff;
    }
    // normalize characters
    if (m_pCharMap) {
        // in-place is fine
        m_InSize = FANormalizeWord (m_pIn, m_InSize, \
            TmpBuff, TmpBuffSize, m_pCharMap);
        m_pIn = TmpBuff;
    }
    // apply transformation, if needed
    if (m_pInTr) {
        ApplyInTr (TmpBuff, TmpBuffSize);
    }

    int Pos = 0;
    int State = m_pDfa->GetInitial ();

    // read word left to right direction
    for (; Pos < m_InSize; ++Pos) {
        const int Iw = (unsigned int) (m_pIn [Pos]);
        State = m_pDfa->GetDest (State, Iw);
        if (-1 == State) {
            break;
        }
    }
    if (-1 != State) {
        const int Ow = m_pState2Ow->GetOw (State);
        // could be -1, if state does not have a reaction
        return Ow;
    }

    return -1;
}


template < class Ty >
const float FAWordToProb_t< Ty >::GetProb (const Ty * pWordStr, const int WordSize)
{
    const int IntProb = GetIntProb (pWordStr, WordSize);

    if (-1 != IntProb) {

        DebugLogAssert (0 <= IntProb && IntProb <= m_MaxIntProb);
        return m_pInt2Float [IntProb];

    } else {

        return float (DefUnfoundProb);
    }
}

}

#endif
