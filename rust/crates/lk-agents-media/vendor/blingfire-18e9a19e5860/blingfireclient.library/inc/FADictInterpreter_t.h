/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_DICTINTERPRETER_T_H_
#define _FA_DICTINTERPRETER_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FADictConfKeeper.h"
#include "FATransformCA_t.h"
#include "FAMphInterpretTools_t.h"
#include "FAArrayCA.h"
#include "FAMultiMapCA.h"
#include "FAState2OwCA.h"
#include "FARSDfaCA.h"
#include "FAMealyDfaCA.h"
#include "FAUtf32Utils.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Dictionary run-time interpreter.
///

template < class Ty >
class FADictInterpreter_t {

public:
    /// creates uninitialized object
    FADictInterpreter_t ();

public:
    /// Note: this method expects initialized configuration object
    /// pInTr can be NULL if no transformation assumed
    void SetConf (
            const FADictConfKeeper * pConf, 
            const FATransformCA_t < Ty > * pInTr
        );

public:
    /// Word -> InfoId, returns -1 if Word does not exist
    const int GetInfoId (
            const Ty * pIn, 
            const int InSize
        ) const;

    /// Word -> Info, returns output size or -1 if Word does not exist
    const int GetInfo (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const;

    /// InfoId -> Info, returns output size or -1 if InfoId does not exist
    const int GetInfo (
            const int InfoId,
            __out_ecount(MaxOutSize) int * pOut,
            const int MaxOutSize
        ) const;

    /// returns maximum info size for the dictionary, returns -1 if the object
    /// is not initialized or MaxSize is not supported
    const int GetMaxInfoSize () const;

private:
    // uses Mealy + Array to get info id (takes normalized word)
    inline const int GetInfoId_mph (const Ty * pIn, const int InSize) const;
    // uses Moore to get info id (takes normalized word)
    inline const int GetInfoId_moore (const Ty * pIn, const int InSize) const;

private:
    // returns object into initial state
    inline void Clear ();

    // makes word normalization, if needed
    inline const int Normalize (
            const Ty * pIn,
            const int InSize,
            __out_ecount(MaxOutSize) Ty * pBuff,
            const int MaxOutSize
        ) const;

private:
    // transformation pointer or NULL
    const FATransformCA_t < Ty > * m_pTr;
    // Moore / Mealy
    int m_FsmType;
    const FARSDfaCA * m_pDfa;
    const FAMealyDfaCA * m_pMealy;
    const FAState2OwCA * m_pState2Ow;
    // K -> I packed array
    const FAArrayCA * m_pK2I;
    // I -> Info multi-map
    const FAMultiMapCA * m_pI2Info;
    // flags
    int m_Direction;
    bool m_IgnoreCase;
    bool m_NoNorm;
    bool m_Ready;
    // MPH tools
    FAMphInterpretTools_t < Ty > m_W2K;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    enum {
        DefDelta = FALimits::MaxWordLen,
    };
};


template < class Ty >
FADictInterpreter_t< Ty >::FADictInterpreter_t () :
    m_pTr (NULL),
    m_FsmType (FAFsmConst::TYPE_MEALY_DFA),
    m_pDfa (NULL),
    m_pMealy (NULL),
    m_pState2Ow (NULL),
    m_pK2I (NULL),
    m_pI2Info (NULL),
    m_Direction (FAFsmConst::DIR_L2R),
    m_IgnoreCase (false),
    m_NoNorm (false),
    m_Ready (false),
    m_pCharMap (NULL)
{
}


template < class Ty >
inline void FADictInterpreter_t< Ty >::Clear ()
{
    m_pTr = NULL;
    m_FsmType = FAFsmConst::TYPE_MEALY_DFA;
    m_pDfa = NULL;
    m_pMealy = NULL;
    m_pState2Ow = NULL;
    m_pK2I = NULL;
    m_pI2Info = NULL;
    m_Direction = FAFsmConst::DIR_L2R;
    m_IgnoreCase = false;
    m_NoNorm = false;
    m_Ready = false;
    m_pCharMap = NULL;
}


template < class Ty >
void FADictInterpreter_t< Ty >::SetConf (
        const FADictConfKeeper * pConf, 
        const FATransformCA_t < Ty > * pInTr
    )
{
    FADictInterpreter_t< Ty >::Clear ();
    DebugLogAssert (false == m_Ready);

    if (NULL == pConf) {
        return;
    }

    m_FsmType = pConf->GetFsmType ();

    if (FAFsmConst::TYPE_MEALY_DFA == m_FsmType) {

        m_pDfa = pConf->GetRsDfa ();
        m_pMealy = pConf->GetMphMealy ();
        m_pK2I = pConf->GetK2I ();

        m_W2K.SetRsDfa (m_pDfa);
        m_W2K.SetMealy (m_pMealy);

        m_Ready = (NULL!=m_pDfa) && (NULL!=m_pMealy) && (NULL!=m_pK2I);

    } else if (FAFsmConst::TYPE_MOORE_DFA == m_FsmType) {

        m_pDfa = pConf->GetRsDfa ();
        m_pState2Ow = pConf->GetState2Ow ();

        m_Ready = (NULL != m_pDfa) && (NULL != m_pState2Ow);
    }

    m_pCharMap = pConf->GetCharMap ();
    m_pI2Info = pConf->GetI2Info ();
    m_Ready = m_Ready && (NULL != m_pI2Info);

    m_Direction = pConf->GetDirection ();
    m_IgnoreCase = pConf->GetIgnoreCase ();

    const bool NoTrUse = pConf->GetNoTrUse ();

    DebugLogAssert (NULL == m_pTr);

    if (false == NoTrUse) {
        m_pTr = pInTr;
    }

    m_NoNorm = (NULL == m_pTr) && (!m_IgnoreCase) && \
        (FAFsmConst::DIR_L2R == m_Direction);
}


template < class Ty >
inline const int FADictInterpreter_t< Ty >::
    Normalize (
        const Ty * pIn,
        const int InSize,
        __out_ecount(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    ) const 
{
    DebugLogAssert (pOut && 0 < MaxOutSize && \
        FALimits::MaxWordSize + DefDelta == MaxOutSize);
    __analysis_assume (pOut && 0 < MaxOutSize && \
        FALimits::MaxWordSize + DefDelta == MaxOutSize);
    DebugLogAssert (0 < InSize && FALimits::MaxWordSize >= InSize && pIn);
    __analysis_assume (0 < InSize && FALimits::MaxWordSize >= InSize && pIn);

    DebugLogAssert (false == m_NoNorm);

    const Ty * pSrc = pIn;
    int SrcSize = InSize;

    // case normalization, if needed
    if (m_IgnoreCase) {

        for (int i = 0 ; i < SrcSize; ++i) {
            const int InSymbol = (int) pSrc [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            pOut [i] = (Ty) OutSymbol ;
        }
        pSrc = pOut;
    }

    // normalize characters
    if (m_pCharMap) {
        // in-place is fine
        SrcSize = FANormalizeWord (pSrc, SrcSize, \
            pOut, MaxOutSize, m_pCharMap);
        pSrc = pOut;
    }

    // apply transformation, if needed
    if (m_pTr) {

        // apply transformation
        const int OutSize = m_pTr->Process (pSrc, SrcSize, pOut, MaxOutSize);

        if (0 < OutSize && MaxOutSize > OutSize) {
            // updated data source, if Tr was successfully applied
            pSrc = pOut;
            SrcSize = OutSize;
        } else if (pOut != pSrc) {
            // copy data if Tr was not applied in-place and it failed
            memcpy (pOut, pSrc, sizeof (Ty) * SrcSize);
        }
    }
    // change the order, if needed (TODO: remove support for AFF order)
    if (FAFsmConst::DIR_R2L == m_Direction) {

        const int SrcSize_2 = SrcSize >> 1;
        for (int i = 0 ; i < SrcSize_2; ++i) {

            const Ty First = pSrc [i];
            const Ty Last = pSrc [SrcSize - i - 1];

            pOut [i] = Last;
            pOut [SrcSize - i - 1] = First;
        }
        pSrc = pOut;
    }

    return SrcSize;
}


template < class Ty >
inline const int FADictInterpreter_t< Ty >::
    GetInfoId_mph (const Ty * pIn, const int InSize) const
{
    DebugLogAssert (0 < InSize && pIn);

    const int K = m_W2K.GetId (pIn, InSize);

    if (-1 == K) {
        return -1;
    }

    DebugLogAssert (m_pK2I);
    DebugLogAssert (0 <= K && K < m_pK2I->GetCount ());

    const int Id = m_pK2I->GetAt (K);
    DebugLogAssert (0 <= Id);

    return Id;
}


template < class Ty >
inline const int FADictInterpreter_t< Ty >::
    GetInfoId_moore (const Ty * pIn, const int InSize) const
{
    DebugLogAssert (0 < InSize && pIn);
    DebugLogAssert (m_pDfa);
    DebugLogAssert (m_pState2Ow);

    int State = m_pDfa->GetInitial ();

    for (int i = 0; i < InSize; ++i) {

        const int Iw = pIn [i];

        State = m_pDfa->GetDest (State, Iw);

        if (-1 == State) {
            return -1;
        }
    }

    const int Id = m_pState2Ow->GetOw (State);
    DebugLogAssert (-1 == Id || m_pDfa->IsFinal (State));

    return Id;
}


template < class Ty >
const int FADictInterpreter_t< Ty >::
    GetInfoId (
        const Ty * pIn, 
        const int InSize
    ) const
{
    const Ty * pIn2;
    int InSize2;

    const int MaxBuffSize = FALimits::MaxWordSize + DefDelta;
    Ty WordBuff [MaxBuffSize];

    if (0 >= InSize || FALimits::MaxWordSize < InSize || !pIn || !m_Ready) {
        return -1;
    }

    if (m_NoNorm) {
        pIn2 = pIn;
        InSize2 = InSize;
    } else {
        pIn2 = WordBuff;
        InSize2 = Normalize (pIn, InSize, WordBuff, MaxBuffSize);
    }

    if (FAFsmConst::TYPE_MOORE_DFA == m_FsmType) {

        const int Id = FADictInterpreter_t::GetInfoId_moore (pIn2, InSize2);
        return Id;

    } else {
        DebugLogAssert (FAFsmConst::TYPE_MEALY_DFA == m_FsmType);

        const int Id = FADictInterpreter_t::GetInfoId_mph (pIn2, InSize2);
        return Id;
    }
}


template < class Ty >
const int FADictInterpreter_t< Ty >::
    GetInfo (
        const Ty * pIn,
        const int InSize,
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize
    ) const
{
    DebugLogAssert (m_Ready);
    DebugLogAssert (m_pI2Info);

    const int Id = FADictInterpreter_t::GetInfoId (pIn, InSize);

    if (-1 == Id) {
        return -1;
    }

    const int OutCount = m_pI2Info->Get (Id, pOut, MaxOutSize);

    return OutCount;
}


template < class Ty >
const int FADictInterpreter_t< Ty >::
    GetInfo (
        const int InfoId,
        __out_ecount(MaxOutSize) int * pOut,
        const int MaxOutSize
    ) const
{
    DebugLogAssert (m_pI2Info);

    if (!m_Ready) {
        return -1;
    }

    const int OutCount = m_pI2Info->Get (InfoId, pOut, MaxOutSize);
    return OutCount;
}


template < class Ty >
const int FADictInterpreter_t< Ty >::GetMaxInfoSize () const
{
    if (!m_Ready) {
        return -1;
    }

    DebugLogAssert (m_pI2Info);

    const int MaxCount = m_pI2Info->GetMaxCount ();
    return MaxCount;
}

}

#endif
