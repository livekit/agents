/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_SUFFIXINTERPRETTOOLS_T_H_
#define _FA_SUFFIXINTERPRETTOOLS_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwsCA.h"
#include "FAMultiMapCA.h"
#include "FATransformCA_t.h"
#include "FAUtf8Utils.h"
#include "FAWftConfKeeper.h"
#include "FASecurity.h"

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
class FASuffixInterpretTools_t {

public:
    FASuffixInterpretTools_t ();
    ~FASuffixInterpretTools_t ();

public:
    /// sets up the configuration
    void SetConf (const FAWftConfKeeper * pConf);
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
        );
    /// returns output string for one-tags transformation rules
    /// returns -1 if no transformation exist
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int Tag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );
    /// returns output string for two-tags transformation rules
    /// returns -1 if no transformation exist
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int FromTag,
            const int ToTag,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

private:
    /// sets up RS-DFA part of automaton
    void SetRsDfa (const FARSDfaCA * pDfa);
    /// sets up State to Ows mapping
    void SetState2Ows (const FAState2OwsCA * pState2Ows);
    /// sets up rules' actions map
    void SetActs (const FAMultiMapCA * pActs);
    /// sets up whether rules have no support for unknown words
    void SetDictMode (const bool DictMode);
    /// sets up whether case normalization should be performed
    void SetIgnoreCase (const bool IgnoreCase);
    /// if necessary normalizes the input, when input is copied into
    /// a temporary buffer, all normalizations are done in-place
    inline void Normalize (
            __out_ecount(MaxOutSize) Ty * pOut, 
            const int MaxOutSize
        );
    // returns the deepest final state starting from StartState
    inline const int GetLastFinal (const int StartState) const;
    // initializes m_pAct and m_ActLen
    inline void GetAction (const int ActNum);
    // returns the output size
    inline const int ApplyAction (
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        ) const;
    // fills in output buffer, returns output size
    inline const int BuildResults (
            const int LastFinal,
            __out_ecount_opt(MaxOutSize) Ty * pOut,
            const int MaxOutSize
        );

private:
    /// input objects
    const FARSDfaCA * m_pDfa;
    const FAState2OwsCA * m_pState2Ows;
    const FAMultiMapCA * m_pActs;
    const FATransformCA_t < Ty > * m_pInTr;
    const FATransformCA_t < Ty > * m_pOutTr;

    /// tmp values
    int m_InitialState;
    const Ty * m_pIn;
    int m_InSize;
    bool m_IgnoreCase;

    /// temporary buffer keeps rule to apply in case of implicit representation
    int * m_pActBuff;
    int m_MaxActBuffSize;

    /// action to be applied
    const int * m_pAct;
    int m_ActLen;

    /// stores here right part numbers to be applied
    int m_MaxOwCount;
    int * m_pOutOws;
    int m_OwCount;

    /// dictionary mode
    bool m_DictMode;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    enum { 
        DefDelimIw = 0,
        DefTmpBuffSize = 1024,
     };
};


template < class Ty >
FASuffixInterpretTools_t< Ty >::
    FASuffixInterpretTools_t () :
        m_pDfa (NULL),
        m_pState2Ows (NULL),
        m_pActs (NULL),
        m_pInTr (NULL),
        m_pOutTr (NULL),
        m_InitialState (-1),
        m_pIn (NULL),
        m_InSize (0),
        m_IgnoreCase (false),
        m_pActBuff (NULL),
        m_MaxActBuffSize (0),
        m_pAct (NULL),
        m_ActLen (0),
        m_MaxOwCount (0),
        m_pOutOws (NULL),
        m_OwCount (0),
        m_DictMode (false),
        m_pCharMap (NULL)
{
}


template < class Ty >
FASuffixInterpretTools_t< Ty >::~FASuffixInterpretTools_t ()
{
    if (m_pOutOws) {
        delete [] m_pOutOws;
        m_pOutOws = NULL;
    }
    if (m_pActBuff) {
        delete [] m_pActBuff;
        m_pActBuff = NULL;
    }
}



template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetConf (const FAWftConfKeeper * pConf)
{
    m_pDfa = NULL;
    m_pState2Ows = NULL;
    m_pActs = NULL;
    m_InitialState = -1;
    m_IgnoreCase = false;
    m_DictMode = false;
    m_pCharMap = NULL;

    const FARSDfaCA * pDfa = pConf->GetRsDfa ();
    const FAState2OwsCA * pState2Ows = pConf->GetState2Ows ();
    const FAMultiMapCA * pActs = pConf->GetActs ();

    // can't be NULL
    LogAssert (pDfa && pState2Ows && pActs);

    // can be NULL
    m_pCharMap = pConf->GetCharMap ();

    const bool DictMode = pConf->GetDictMode ();
    const bool IgnoreCase = pConf->GetIgnoreCase ();
    const bool UseNfst = pConf->GetUseNfst ();

    // can't be set here
    LogAssert (!UseNfst);

    SetRsDfa (pDfa);
    SetState2Ows (pState2Ows);
    SetActs (pActs);
    SetDictMode (DictMode);
    SetIgnoreCase (IgnoreCase);
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetIgnoreCase (const bool IgnoreCase)
{
    m_IgnoreCase = IgnoreCase;
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetRsDfa (const FARSDfaCA * pDfa)
{
    m_pDfa = pDfa;

    if (m_pDfa) {
        m_InitialState = m_pDfa->GetInitial ();
        LogAssert (-1 != m_InitialState);
    }
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetState2Ows (const FAState2OwsCA * pState2Ows)
{
    m_pState2Ows = pState2Ows;

    if (NULL != m_pState2Ows) {

        m_MaxOwCount = m_pState2Ows->GetMaxOwsCount ();
        LogAssert (0 < m_MaxOwCount);

        if (m_pOutOws) {
            delete [] m_pOutOws;
            m_pOutOws = NULL;
        }

        m_pOutOws = NEW int [m_MaxOwCount];
        LogAssert (m_pOutOws);
    }
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetActs (const FAMultiMapCA * pActs)
{
    m_pActs = pActs;

    if (NULL != m_pActs) {

        m_MaxActBuffSize = m_pActs->GetMaxCount ();
        LogAssert (0 < m_MaxActBuffSize);

        if (m_pActBuff) {
            delete [] m_pActBuff;
            m_pActBuff = NULL;
        }

        m_pActBuff = NEW int [m_MaxActBuffSize];
        LogAssert (m_pActBuff);
    }
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetInTr (const FATransformCA_t < Ty > * pInTr)
{
    m_pInTr = pInTr;
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetOutTr (const FATransformCA_t < Ty > * pOutTr)
{
    m_pOutTr = pOutTr;
}


template < class Ty >
void FASuffixInterpretTools_t< Ty >::
    SetDictMode (const bool DictMode)
{
    m_DictMode = DictMode;
}


template < class Ty >
inline const int FASuffixInterpretTools_t< Ty >::
    GetLastFinal (const int StartState) const
{
    DebugLogAssert (0 < m_InSize && m_pIn);

    int LastFinal = -1;
    int State = StartState;

    if (false == m_DictMode) {

        /// find the longest match
        for (int Pos = m_InSize - 1; Pos >= 0; --Pos) {

            const int Iw = (unsigned int) (m_pIn [Pos]);
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
        for (int Pos = m_InSize - 1; Pos >= 0; --Pos) {

            const int Iw = (unsigned int) (m_pIn [Pos]);
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
inline void FASuffixInterpretTools_t< Ty >::
    GetAction (const int ActNum)
{
    DebugLogAssert (m_pActs);

    // see whether there are some explicitly stored data
    m_ActLen = m_pActs->Get (ActNum, &m_pAct);

    // get implicitly stored data
    if (-1 == m_ActLen) {
        m_ActLen = m_pActs->Get (ActNum, m_pActBuff, m_MaxActBuffSize);
        DebugLogAssert (-1 != m_ActLen && m_ActLen <= m_MaxActBuffSize);
        m_pAct = m_pActBuff;
    }
}


template < class Ty >
inline const int FASuffixInterpretTools_t< Ty >::
    ApplyAction (
        __out_ecount_opt(MaxOutSize) Ty * pOut,
        const int MaxOutSize
    ) const
{
    DebugLogAssert (0 < m_InSize && m_pIn);
    DebugLogAssert (0 < m_ActLen && m_pAct);

    int i, PrefLen, SuffLen, PrefCut, SuffCut;
    const int * pPref;
    const int * pSuff;

    /// decode the action
    if (FALimits::MaxWordSize >= *m_pAct) {
        // no prefix data
        pPref = NULL;
        PrefLen = 0;
        PrefCut = 0;
        // get suffix data
        pSuff = m_pAct + 1;
        SuffCut = *m_pAct;
        SuffLen = m_ActLen - 1;
    } else {
        // get skip value
        const int Skip = *m_pAct - FALimits::MaxWordSize;
        DebugLogAssert (1 < Skip && Skip <= FALimits::MaxWordSize);
        // get prefix data
        pPref = m_pAct + 2;
        PrefCut = m_pAct [1];
        PrefLen = Skip - 2;
        // get suffix data
        pSuff = m_pAct + Skip + 1;
        SuffCut = m_pAct [Skip];
        SuffLen = m_ActLen - Skip - 1;
    }

    DebugLogAssert (0 <= PrefLen && PrefLen <= FALimits::MaxWordSize);
    DebugLogAssert (0 <= SuffLen && SuffLen <= FALimits::MaxWordSize);
    DebugLogAssert (0 <= PrefCut && PrefCut <= FALimits::MaxWordSize);
    DebugLogAssert (0 <= SuffCut && SuffCut <= FALimits::MaxWordSize);

    /// calc the common length
    const int CommonLen = m_InSize - SuffCut - PrefCut;
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
        memcpy (pCurrOut, m_pIn + PrefCut, CommonLen * sizeof (Ty));
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
inline const int FASuffixInterpretTools_t< Ty >::
    BuildResults (
        const int LastFinal,
        __out_ecount_opt(MaxOutSize) Ty * pOut, 
        const int MaxOutSize
    )
{
    DebugLogAssert (-1 != LastFinal);

    m_OwCount = m_pState2Ows->GetOws (LastFinal, m_pOutOws, m_MaxOwCount);
    DebugLogAssert (m_OwCount <= m_MaxOwCount);

    if (0 < m_OwCount) {

        int OutSize = 0;

        for (int i = 0; i < m_OwCount; ++i) {

            const int ActNum = m_pOutOws [i];
            Ty * pCurrOut = pOut + OutSize;
            const int CurrMaxSize = MaxOutSize - OutSize;

            GetAction (ActNum);

            OutSize += ApplyAction (pCurrOut, CurrMaxSize);

        } // of for (int i = 0; ...

        return OutSize;

    } // of if (0 < m_OwCount) ...

    return -1;
}


template < class Ty >
inline void FASuffixInterpretTools_t< Ty >::
    Normalize (__out_ecount(MaxOutSize) Ty * pOut, const int MaxOutSize)
{
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    DebugLogAssert (pOut && 0 < MaxOutSize);
    __analysis_assume (pOut && 0 < MaxOutSize);
    DebugLogAssert (MaxOutSize >= m_InSize);
    __analysis_assume (MaxOutSize >= m_InSize);

    // normalize the case, if needed
    if (m_IgnoreCase) {
        for (int i = 0 ; i < m_InSize; ++i) {
            const int InSymbol = (int) m_pIn [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            pOut [i] = (Ty) OutSymbol ;
        }
        m_pIn = pOut;
    }

    // normalize characters
    if (m_pCharMap) {
        // in-place is fine
        m_InSize = FANormalizeWord (m_pIn, m_InSize, \
            pOut, MaxOutSize, m_pCharMap);
        m_pIn = pOut;
    }

    // apply the transformation, if needed
    if (m_pInTr) {

        // in-place is fine
        const int OutSize = \
            m_pInTr->Process (m_pIn, m_InSize, pOut, MaxOutSize);

        // see whether there was some transformation made
        if (0 < OutSize && OutSize <= MaxOutSize) {
            m_pIn = pOut;
            m_InSize = OutSize;
        }
    }
}


template < class Ty >
const int FASuffixInterpretTools_t< Ty >::
    Process (const Ty * pIn,
             const int InSize,
             __out_ecount_opt(MaxOutSize) Ty * pOut,
             const int MaxOutSize)
{
    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];

    m_pIn = pIn;
    m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    __analysis_assume (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);

    Normalize (TmpBuff, TmpBuffSize);

    const int LastFinal = GetLastFinal (m_InitialState);

    if (-1 != LastFinal) {

        const int OutSize = BuildResults (LastFinal, pOut, MaxOutSize);
        return OutSize;
    }

    return -1;
}


template < class Ty >
const int FASuffixInterpretTools_t< Ty >::
    Process (const Ty * pIn,
             const int InSize,
             const int Tag,
             __out_ecount_opt(MaxOutSize) Ty * pOut,
             const int MaxOutSize)
{
    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];

    m_pIn = pIn;
    m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    __analysis_assume (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);

    Normalize (TmpBuff, TmpBuffSize);

    // process Tag
    int State = m_pDfa->GetDest (m_InitialState, Tag);
    if (-1 == State) {
        return -1;
    }
    // calc the last final state
    int LastFinal = GetLastFinal (State);
    if (-1 == LastFinal && m_pDfa->IsFinal (State)) {
        LastFinal = State;
    }
    // get actions and apply them
    if (-1 != LastFinal) {

        const int OutSize = BuildResults (LastFinal, pOut, MaxOutSize);
        return OutSize;
    }

    return -1;
}


template < class Ty >
const int FASuffixInterpretTools_t< Ty >::
    Process (const Ty * pIn,
             const int InSize,
             const int FromTag,
             const int ToTag,
             __out_ecount_opt(MaxOutSize) Ty * pOut,
             const int MaxOutSize)
{
    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];

    m_pIn = pIn;
    m_InSize = InSize;

    DebugLogAssert (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    __analysis_assume (m_pDfa && m_pState2Ows && m_pActs && -1 != m_InitialState);
    DebugLogAssert (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);
    __analysis_assume (m_pIn && 0 < m_InSize && m_InSize <= FALimits::MaxWordLen);

    Normalize (TmpBuff, TmpBuffSize);

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

        const int NewLastFinal = GetLastFinal (State);

        if (-1 != NewLastFinal) {

            LastFinal = NewLastFinal;

        } else if (m_pDfa->IsFinal (State)) {

            LastFinal = State;
        }
    }
    // get actions and apply them
    if (-1 != LastFinal) {

        const int OutSize = BuildResults (LastFinal, pOut, MaxOutSize);
        return OutSize;
    }

    return -1;
}

}

#endif
