/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SUFFIXINTERPRETTOOLSCONST_T_H_
#define _FA_SUFFIXINTERPRETTOOLSCONST_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwsCA.h"
#include "FAMultiMapCA.h"
#include "FATransformCA_t.h"
#include "FAUtf8Utils.h"
#include "FAWftConfKeeper.h"
#include "FASecurity.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

///
/// Suffix rules interpreter. Converts Word, Word/Tag or Word/TagFrom/TagTo
/// to whatever rules for. See FASuffixRules2Chains.h for syntax description.
///
/// Notes: 
///
/// 1. All Process methods return 0-separated word list. If the output
/// buffer is not large enough then it just returns the minimum required size.
/// 2. Input transformation, if supplied, makes output to a temporary buffer.
/// Output transformation, if supplied, makes transformation in-place.
///

template < class Ty >
class FASuffixInterpretToolsConst_t {

public:
    FASuffixInterpretToolsConst_t ();
    ~FASuffixInterpretToolsConst_t ();

public:
    /// sets up the configuration
    void SetConf (const FAWftConfKeeper * pConf, FAAllocatorA * pMemMgr);
    /// sets up input transformation, can be NULL
    void SetInTr (const FATransformCA_t < Ty > * pInTr);
    /// sets up output transformation, can be NULL
    void SetOutTr (const FATransformCA_t < Ty > * pOutTr);

    /// returns output string for tagless rules
    /// returns -1 if no transformation exist
    const int Process (
            const Ty * pIn, 
            const int InSize, 
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    /// returns output string for one-tags transformation rules
    /// returns -1 if no transformation exist
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int Tag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    /// returns output string for two-tags transformation rules
    /// returns -1 if no transformation exist
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;

private:
    struct Scratch {
        /// input sequence
        const Ty * m_pIn;
        int m_InSize;
        /// temporary buffer keeps rule to apply in case of implicit representation
        int * m_pActBuff;
        /// action to be applied
        const int * m_pAct;
        int m_ActLen;
        /// rule ids to be applied
        int * m_pOutOws;
        int m_OwCount;
        /// a memlfp action buffer
        FAArray_cont_t < int > m_act;
        /// a memlfp ows buffer
        FAArray_cont_t < int > m_ows;
    };

private:
    /// initializes a new Scratch object
    inline void InitializeScratch (Scratch * pTmp) const;
    /// if necessary normalizes the input, when input is copied into
    /// a temporary buffer, all normalizations are done in-place
    inline void Normalize (
            Scratch * pTmp,
            __out_ecount(MaxOutSize) Ty * pOut, 
            const int MaxOutSize
        ) const;
    // returns the deepest final state starting from StartState
    inline const int GetLastFinal (const Scratch * pTmp, const int StartState) const;
    // initializes m_pAct and m_ActLen
    inline void GetAction (Scratch * pTmp, const int ActNum) const;
    // fills in output buffer, returns output size
    inline const int BuildResults (
            Scratch * pTmp, 
            const int LastFinal,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    // returns the output size
    inline const int ApplyAction (
            const Scratch * pTmp,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;

private:
    /// input objects
    const FARSDfaCA * m_pDfa;
    const FAState2OwsCA * m_pState2Ows;
    const FAMultiMapCA * m_pActs;
    const FATransformCA_t < Ty > * m_pInTr;
    const FATransformCA_t < Ty > * m_pOutTr;

    /// configuration related variables
    int m_InitialState;
    bool m_IgnoreCase;
    int m_MaxActBuffSize;
    int m_MaxOwCount;
    bool m_DictMode;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    /// memory manager
    FAAllocatorA * m_pMemMgr;

    enum { 
        DefDelimIw = 0,
        DefTmpBuffSize = 1024,
     };
};


template < class Ty >
FASuffixInterpretToolsConst_t< Ty >::
    FASuffixInterpretToolsConst_t () :
        m_pDfa (NULL),
        m_pState2Ows (NULL),
        m_pActs (NULL),
        m_pInTr (NULL),
        m_pOutTr (NULL),
        m_InitialState (-1),
        m_IgnoreCase (false),
        m_MaxActBuffSize (0),
        m_MaxOwCount (0),
        m_DictMode (false),
        m_pCharMap (NULL),
        m_pMemMgr (NULL)
{
}


template < class Ty >
FASuffixInterpretToolsConst_t< Ty >::~FASuffixInterpretToolsConst_t ()
{
}


template < class Ty >
void FASuffixInterpretToolsConst_t< Ty >::
    SetConf (const FAWftConfKeeper * pConf, FAAllocatorA * pMemMgr)
{
    LogAssert (pConf && pMemMgr);

    m_pMemMgr = pMemMgr;

    m_pDfa = NULL;
    m_pState2Ows = NULL;
    m_pActs = NULL;
    m_InitialState = -1;
    m_IgnoreCase = false;
    m_DictMode = false;
    m_pCharMap = NULL;

    m_pDfa = pConf->GetRsDfa ();
    m_pState2Ows = pConf->GetState2Ows ();
    m_pActs = pConf->GetActs ();

    // can't be NULL
    LogAssert (m_pDfa && m_pState2Ows && m_pActs);

    m_pCharMap = pConf->GetCharMap ();
    m_DictMode = pConf->GetDictMode ();
    m_IgnoreCase = pConf->GetIgnoreCase ();
    const bool UseNfst = pConf->GetUseNfst ();
    // can't be set here
    LogAssert (!UseNfst);

    // get the initial state
    m_InitialState = m_pDfa->GetInitial ();
    LogAssert (-1 != m_InitialState);

    // get the maximum possible amount of Ows
    m_MaxOwCount = m_pState2Ows->GetMaxOwsCount ();
    LogAssert (0 < m_MaxOwCount);

    // get maximum possible action size
    m_MaxActBuffSize = m_pActs->GetMaxCount ();
    LogAssert (0 < m_MaxActBuffSize);
}


template < class Ty >
void FASuffixInterpretToolsConst_t< Ty >::
    SetInTr (const FATransformCA_t < Ty > * pInTr)
{
    m_pInTr = pInTr;
}


template < class Ty >
void FASuffixInterpretToolsConst_t< Ty >::
    SetOutTr (const FATransformCA_t < Ty > * pOutTr)
{
    m_pOutTr = pOutTr;
}


template < class Ty >
inline void FASuffixInterpretToolsConst_t< Ty >::
    InitializeScratch (Scratch * pTmp) const
{
    DebugLogAssert (pTmp);
    DebugLogAssert (0 < m_MaxOwCount);
    DebugLogAssert (0 < m_MaxActBuffSize);

    pTmp->m_ows.SetAllocator (m_pMemMgr);
    pTmp->m_ows.Clear ();
    pTmp->m_ows.resize (m_MaxOwCount);
    pTmp->m_pOutOws = pTmp->m_ows.begin ();
    LogAssert (pTmp->m_pOutOws);

    pTmp->m_act.SetAllocator (m_pMemMgr);
    pTmp->m_act.Clear ();
    pTmp->m_act.resize (m_MaxActBuffSize);
    pTmp->m_pActBuff = pTmp->m_act.begin ();
    LogAssert (pTmp->m_pActBuff);
}


template < class Ty >
inline const int FASuffixInterpretToolsConst_t< Ty >::
    GetLastFinal (const Scratch * pTmp, const int StartState) const
{
    const Ty * pIn = pTmp->m_pIn;
    const int InSize = pTmp->m_InSize;
    DebugLogAssert (0 < InSize && pIn);

    int LastFinal = -1;
    int State = StartState;

    if (false == m_DictMode) {

        /// find the longest match
        for (int Pos = InSize - 1; Pos >= 0; --Pos) {

            const int Iw = (unsigned int) (pIn [Pos]);
            State = m_pDfa->GetDest (State, Iw);
            if (-1 == State) {
                break;
            }
            if (m_pDfa->IsFinal (State)) {
                LastFinal = State;
            }
        }
        /// try to make one more step by a delimiter
        if (-1 != State) {
            State = m_pDfa->GetDest (State, DefDelimIw);
            if (-1 != State && m_pDfa->IsFinal (State)) {
                LastFinal = State;
            }
        }

    } else {

        /// match the whole input
        for (int Pos = InSize - 1; Pos >= 0; --Pos) {

            const int Iw = (unsigned int) (pIn [Pos]);
            State = m_pDfa->GetDest (State, Iw);
            if (-1 == State) {
                break;
            }
        }
        if (-1 != State && m_pDfa->IsFinal (State)) {
            LastFinal = State;
        }

    } // of if (false == m_DictMode) ...

    return LastFinal;
}


template < class Ty >
inline void FASuffixInterpretToolsConst_t< Ty >::
    GetAction (Scratch * pTmp, const int ActNum) const
{
    DebugLogAssert (m_pActs);

    // see whether there are some explicitly stored data
    pTmp->m_ActLen = m_pActs->Get (ActNum, &pTmp->m_pAct);

    // get implicitly stored data
    if (-1 == pTmp->m_ActLen) {
        pTmp->m_ActLen = m_pActs->Get (ActNum, pTmp->m_pActBuff, m_MaxActBuffSize);
        DebugLogAssert (-1 != pTmp->m_ActLen && pTmp->m_ActLen <= m_MaxActBuffSize);
        pTmp->m_pAct = pTmp->m_pActBuff;
    }
}


template < class Ty >
inline const int FASuffixInterpretToolsConst_t< Ty >::
    ApplyAction (
        const Scratch * pTmp,
        __out_ecount_opt(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    ) const
{
    DebugLogAssert (pTmp);

    // get the action to be applied
    const int * pAct = pTmp->m_pAct;
    const int ActLen = pTmp->m_ActLen;
    DebugLogAssert (0 < ActLen && pAct);

    // get the input 
    const Ty * pIn = pTmp->m_pIn;
    const int InSize = pTmp->m_InSize;
    DebugLogAssert (0 < InSize && pIn);

    int i, PrefLen, SuffLen, PrefCut, SuffCut;
    const int * pPref;
    const int * pSuff;

    /// decode the action
    if (FALimits::MaxWordSize >= *pAct) {
        // no prefix data
        pPref = NULL;
        PrefLen = 0;
        PrefCut = 0;
        // get suffix data
        pSuff = pAct + 1;
        SuffCut = *pAct;
        SuffLen = ActLen - 1;
    } else {
        // get skip value
        const int Skip = *pAct - FALimits::MaxWordSize;
        DebugLogAssert (1 < Skip && Skip <= FALimits::MaxWordSize);
        // get prefix data
        pPref = pAct + 2;
        PrefCut = pAct [1];
        PrefLen = Skip - 2;
        // get suffix data
        pSuff = pAct + Skip + 1;
        SuffCut = pAct [Skip];
        SuffLen = ActLen - Skip - 1;
    }

    DebugLogAssert (0 <= PrefLen && PrefLen <= FALimits::MaxWordSize);
    DebugLogAssert (0 <= SuffLen && SuffLen <= FALimits::MaxWordSize);
    DebugLogAssert (0 <= PrefCut && PrefCut <= FALimits::MaxWordSize);
    DebugLogAssert (0 <= SuffCut && SuffCut <= FALimits::MaxWordSize);

    /// calc the common length
    const int CommonLen = InSize - SuffCut - PrefCut;
    if (0 > CommonLen || 0 > SuffLen || 0 > PrefLen || \
        (0 == CommonLen && 0 == SuffLen && 0 == PrefLen)) {
        return 0;
    }
    /// calc the output size + terminating 0
    int OutSize = CommonLen + SuffLen + PrefLen + 1;
    if (OutSize > MaxOutSize) {
        return OutSize;
    }

    DebugLogAssert (pOut);
    Ty * pCurrOut = pOut;

    /// add a new prefix, if needed
    for (i = 0; i < PrefLen; ++i) {
        pCurrOut [i] = (Ty) pPref [i];
    }
    pCurrOut += PrefLen;
    /// copy a common part as-is
    if (0 < CommonLen) {
        memcpy (pCurrOut, pIn + PrefCut, CommonLen * sizeof (Ty));
        pCurrOut += CommonLen;
    }
    /// add a new suffix, if needed
    for (i = 0; i < SuffLen; ++i) {
        pCurrOut [i] = (Ty) pSuff [i];
    }
    pCurrOut += SuffLen;
    /// apply out-transformation in-place
    if (m_pOutTr) {

        // uses MaxOutSize - 1 as we'll need the last position for 0
        const int NewLen = \
            m_pOutTr->Process (pOut, OutSize - 1, pOut, MaxOutSize - 1);

        if (-1 != NewLen && NewLen < MaxOutSize) {
            OutSize = NewLen + 1;
            pCurrOut = pOut + NewLen;
        }
    }
    /// add terminating zero, generated word cannot be empty here
    *pCurrOut = 0;
    return OutSize;
}


template < class Ty >
inline const int FASuffixInterpretToolsConst_t< Ty >::
    BuildResults (
        Scratch * pTmp,
        const int LastFinal,
        __out_ecount_opt(MaxOutSize) Ty * pOut, 
        const int MaxOutSize
    ) const
{
    DebugLogAssert (-1 != LastFinal);

    pTmp->m_OwCount = m_pState2Ows->GetOws (LastFinal, pTmp->m_pOutOws, m_MaxOwCount);
    DebugLogAssert (pTmp->m_OwCount <= m_MaxOwCount);

    if (0 < pTmp->m_OwCount) {

        int OutSize = 0;

        for (int i = 0; i < pTmp->m_OwCount; ++i) {

            const int ActNum = pTmp->m_pOutOws [i];
            Ty * pCurrOut = pOut + OutSize;
            const int CurrMaxSize = MaxOutSize - OutSize;

            GetAction (pTmp, ActNum);

            OutSize += ApplyAction (pTmp, pCurrOut, CurrMaxSize);

        } // of for (int i = 0; ...

        return OutSize;

    } // of if (0 < pTmp->m_OwCount) ...

    return -1;
}


template < class Ty >
inline void FASuffixInterpretToolsConst_t< Ty >::
    Normalize (Scratch * pTmp, __out_ecount(MaxOutSize) Ty * pOut, const int MaxOutSize) const
{
    DebugLogAssert (pTmp->m_pIn && 0 < pTmp->m_InSize && pTmp->m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (pTmp->m_pIn && 0 < pTmp->m_InSize && pTmp->m_InSize <= FALimits::MaxWordLen);
    DebugLogAssert (pOut && 0 < MaxOutSize);
    __analysis_assume (pOut && 0 < MaxOutSize);
    DebugLogAssert (MaxOutSize >= pTmp->m_InSize);
    __analysis_assume (MaxOutSize >= pTmp->m_InSize);

    // normalize the case, if needed
    if (m_IgnoreCase) {
        for (int i = 0 ; i < pTmp->m_InSize; ++i) {
            const int InSymbol = (int) pTmp->m_pIn [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            pOut [i] = (Ty) OutSymbol ;
        }
        pTmp->m_pIn = pOut;
    }

    // normalize characters
    if (m_pCharMap) {
        // in-place is fine
        pTmp->m_InSize = FANormalizeWord (pTmp->m_pIn, pTmp->m_InSize, \
            pOut, MaxOutSize, m_pCharMap);
        pTmp->m_pIn = pOut;
    }

    // apply the transformation, if needed
    if (m_pInTr) {
        // in-place is fine
        const int OutSize = \
            m_pInTr->Process (pTmp->m_pIn, pTmp->m_InSize, pOut, MaxOutSize);

        // see whether there was some transformation made
        if (0 < OutSize && OutSize <= MaxOutSize) {
            pTmp->m_pIn = pOut;
            pTmp->m_InSize = OutSize;
        }
    }
}


template < class Ty >
const int FASuffixInterpretToolsConst_t< Ty >::
    Process (const Ty * pIn,
             const int InSize,
             __out_ecount_opt(MaxOutSize) Ty * pOut,
             const int MaxOutSize) const
{
    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];
    Scratch tmp;
    
    InitializeScratch (&tmp);

    tmp.m_pIn = pIn;
    tmp.m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    __analysis_assume (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    DebugLogAssert (pIn && 0 < InSize && InSize <= FALimits::MaxWordLen);
    __analysis_assume (pIn && 0 < InSize && InSize <= FALimits::MaxWordLen);

    Normalize (&tmp, TmpBuff, TmpBuffSize);

    const int LastFinal = GetLastFinal (&tmp, m_InitialState);

    if (-1 != LastFinal) {

        const int OutSize = BuildResults (&tmp, LastFinal, pOut, MaxOutSize);
        return OutSize;
    }

    return -1;
}


template < class Ty >
const int FASuffixInterpretToolsConst_t< Ty >::
    Process (const Ty * pIn,
             const int InSize,
             const int Tag,
             __out_ecount_opt(MaxOutSize) Ty * pOut,
             const int MaxOutSize) const
{
    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];
    Scratch tmp;
    
    InitializeScratch (&tmp);

    tmp.m_pIn = pIn;
    tmp.m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    __analysis_assume (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    DebugLogAssert (pIn && 0 < InSize && InSize <= FALimits::MaxWordLen);
    __analysis_assume (pIn && 0 < InSize && InSize <= FALimits::MaxWordLen);

    Normalize (&tmp, TmpBuff, TmpBuffSize);

    // process Tag
    int State = m_pDfa->GetDest (m_InitialState, Tag);
    if (-1 == State) {
        return -1;
    }
    // calc the last final state
    int LastFinal = GetLastFinal (&tmp, State);
    if (-1 == LastFinal && m_pDfa->IsFinal (State)) {
        LastFinal = State;
    }
    // get actions and apply them
    if (-1 != LastFinal) {

        const int OutSize = BuildResults (&tmp, LastFinal, pOut, MaxOutSize);
        return OutSize;
    }

    return -1;
}


template < class Ty >
const int FASuffixInterpretToolsConst_t< Ty >::
    Process (const Ty * pIn,
             const int InSize,
             const int FromTag,
             const int ToTag,
             __out_ecount_opt(MaxOutSize) Ty * pOut,
             const int MaxOutSize) const
{
    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];
    Scratch tmp;
    
    InitializeScratch (&tmp);

    tmp.m_pIn = pIn;
    tmp.m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    __analysis_assume (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    DebugLogAssert (pIn && 0 < InSize && InSize <= FALimits::MaxWordLen);
    __analysis_assume (pIn && 0 < InSize && InSize <= FALimits::MaxWordLen);

    Normalize (&tmp, TmpBuff, TmpBuffSize);

    int LastFinal = -1;

    // process FromTag
    int State = m_pDfa->GetDest (m_InitialState, FromTag);
    if (-1 == State) {
        return -1;
    }
    if (m_pDfa->IsFinal (State)) {
        LastFinal = State;
    }
    // process ToTag
    State = m_pDfa->GetDest (State, ToTag);

    if (-1 != State) {

        const int NewLastFinal = GetLastFinal (&tmp, State);

        if (-1 != NewLastFinal) {

            LastFinal = NewLastFinal;

        } else if (m_pDfa->IsFinal (State)) {

            LastFinal = State;
        }
    }
    // get actions and apply them
    if (-1 != LastFinal) {

        const int OutSize = BuildResults (&tmp, LastFinal, pOut, MaxOutSize);
        return OutSize;
    }

    return -1;
}

}

#endif
