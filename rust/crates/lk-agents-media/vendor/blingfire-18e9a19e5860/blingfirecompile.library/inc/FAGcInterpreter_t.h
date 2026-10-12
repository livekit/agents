/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_GCINTERPRETER_T_H_
#define _FA_GCINTERPRETER_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAArray_cont_t.h"
#include "FAGcLDB.h"
#include "FAActionsA.h"
#include "FABrResult.h"
#include "FADigitizer_t.h"
#include "FADigitizer_dct_t.h"
#include "FAAutInterpretTools2_trbr_t.h"
#include "FAResolveMatchA.h"
#include "FAState2OwsCA.h"

namespace BlingFire
{

class FAAllocatorA;

///
/// Left part interpreter for left-hand-side GC-like rules. Takes single dump
/// (dump array) file as a data input. Ty defines word character type.
///
/// Usage:
///   1. SetLDB
///   2. SetActions
///   3. SetIgnoreCase
///   foreach (Word in Input) do
///     4. AddWord (Word)
///     5. Process ()
///   done
///

template < class Ty >
class FAGcInterpreter_t {

public:
    FAGcInterpreter_t (FAAllocatorA * pAlloc);

public:
    /// sets up all resources needed for interpretation
    void SetLDB (const FAGcLDB * pLDB);
    /// sets up action execution call-back
    void SetActions (const FAActionsA * pActs);
    /// sets up ignore-case flag
    void SetIgnoreCase (const bool IgnoreCase);
    /// sets up match-resolver, if not specified uses all match-results
    void SetResolver (FAResolveMatchA * pResolver);
    /// sets up tag dictionary (needed for Process1 only)
    void SetTagDict (const FADictInterpreter_t <Ty> * pTagDict);
    /// adds next word (object should already be set up)
    void AddWord (
            const Ty * pText,       // word text
            const int TextLen,      // word text length
            const int Tag,          // POS tag after disambiguation
            const int DctSetId = -1 // tag dictionary id for the given word
        );
    /// sets up the context which will passed to actions, NULL by default
    void SetContext (void * pContext);
    /// makes analysis
    void Process ();

private:
    inline void Prepare ();
    inline void Process_int ();
    inline const int UpdateResults (
            const int From, 
            const int To, 
            const int State
        );
    inline const int UpdateResults_resolver (
            const int From, 
            const int To, 
            const int State
        );
    inline void ApplyResults_resolver ();
    inline void Clear ();

private:
    /// LDB
    const FAGcLDB * m_pLDB;
    int m_TokenType;
    int m_TagOwBase;
    const FAActionsA * m_pActs;
    int m_WordCount;
    /// text digitizer
    FADigitizer_t < Ty > m_w2ow;
    /// dict digitizer
    FADigitizer_dct_t < Ty > m_w2ow_dct;
    /// contiguous storage for tuples: <iw1, iw2>_1, <iw1, iw2>_2 ...
    FAArray_cont_t < int > m_i2ows;
    const int * m_pI2Ows;
    /// common rules automaton
    const FARSDfaCA * m_pRulesDfa;
    const FAState2OwsCA * m_pRulesMap;
    /// storage for matched rule nums
    FAArray_cont_t < int > m_rule_nums;
    int * m_pRuleNums;
    int m_MaxRuleCount;
    /// optional match resolver
    FAResolveMatchA * m_pResolver;
    /// trbr interpreter
    FAAutInterpretTools2_trbr_t < int > m_r2trbr;
    /// sub-expressions
    FABrResult m_trbrs;
    /// context pointer, can be NULL
    void * m_pContext;
};


template < class Ty >
FAGcInterpreter_t< Ty >::FAGcInterpreter_t (FAAllocatorA * pAlloc) :
    m_pLDB (NULL),
    m_TokenType (0),
    m_TagOwBase (0),
    m_pActs (NULL),
    m_WordCount (1),
    m_pI2Ows (NULL),
    m_pRulesDfa (NULL),
    m_pRulesMap (NULL),
    m_pRuleNums (NULL),
    m_MaxRuleCount (0),
    m_pResolver (NULL),
    m_r2trbr (pAlloc),
    m_trbrs (pAlloc),
    m_pContext (NULL)
{
    m_i2ows.SetAllocator (pAlloc);
    m_i2ows.Create ();
    m_i2ows.resize (FAFsmConst::DIGITIZER_COUNT);

    int * pOut = m_i2ows.begin ();
    for (int i = 0; i < FAFsmConst::DIGITIZER_COUNT; ++i) {
        pOut [i] = FAFsmConst::IW_L_ANCHOR;
    }

    m_rule_nums.SetAllocator (pAlloc);
    m_rule_nums.Create ();

    m_w2ow.SetAnyIw (FAFsmConst::IW_ANY);
    m_w2ow.SetAnyOw (FAFsmConst::IW_ANY);

    m_w2ow_dct.SetAnyOw (FAFsmConst::IW_ANY);
}


template < class Ty >
void FAGcInterpreter_t< Ty >::SetIgnoreCase (const bool IgnoreCase)
{
    m_w2ow.SetIgnoreCase (IgnoreCase);
}


template < class Ty >
void FAGcInterpreter_t< Ty >::SetLDB (const FAGcLDB * pLDB)
{
    m_TokenType = 0;
    m_TagOwBase = 0;
    m_pRulesDfa = NULL;
    m_pRulesMap = NULL;
    m_pRuleNums = NULL;
    m_pLDB = pLDB;

    if (pLDB) {

        const FAWREConfCA * pCm = pLDB->GetCommon ();
        DebugLogAssert (pCm);

        m_TokenType = pCm->GetTokenType ();
        m_TagOwBase = pCm->GetTagOwBase ();

        m_pRulesDfa = pCm->GetDfa1 ();
        DebugLogAssert (m_pRulesDfa);
        m_pRulesMap = pCm->GetState2Ows ();
        DebugLogAssert (m_pRulesMap);
        m_MaxRuleCount = m_pRulesMap->GetMaxOwsCount ();
        DebugLogAssert (0 < m_MaxRuleCount);

        m_rule_nums.resize (m_MaxRuleCount);
        m_pRuleNums = m_rule_nums.begin ();

        if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {

            const FARSDfaCA * pDigDfa = pCm->GetTxtDigDfa ();
            const FAState2OwCA * pDigMap = pCm->GetTxtDigOws ();

            DebugLogAssert (pDigDfa && pDigMap);

            m_w2ow.SetRsDfa (pDigDfa);
            m_w2ow.SetState2Ow (pDigMap);
            m_w2ow.Prepare ();
        }
        if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {

            const FAArrayCA * pSet2Ow = pCm->GetDictDig ();
            DebugLogAssert (pSet2Ow);

            m_w2ow_dct.SetSet2Ow (pSet2Ow);
        }
    }
}


template < class Ty >
void FAGcInterpreter_t< Ty >::SetActions (const FAActionsA * pActs)
{
    m_pActs = pActs;
}


template < class Ty >
void FAGcInterpreter_t< Ty >::SetResolver (FAResolveMatchA * pResolver)
{
    m_pResolver = pResolver;
}


template < class Ty >
void FAGcInterpreter_t< Ty >::
    SetTagDict (const FADictInterpreter_t <Ty> * pTagDict)
{
    m_w2ow_dct.SetTagDict (pTagDict);
}


template < class Ty >
void FAGcInterpreter_t< Ty >::SetContext (void * pContext)
{
    m_pContext = pContext;
}


template < class Ty >
void FAGcInterpreter_t< Ty >::
    AddWord (
        const Ty * pText, 
        const int TextLen, 
        const int Tag, 
        const int DctSetId
    )
{
    DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);
    DebugLogAssert (pText && 0 < TextLen);

    int Ow;

    // get Ow from text digitizer
    if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {

        Ow = m_w2ow.Process (pText, TextLen);
        m_i2ows.push_back (Ow);

    } else {
        m_i2ows.push_back (-1);
    }
    // get Ow from tag digitizer
    if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {

        Ow = Tag + m_TagOwBase;
        m_i2ows.push_back (Ow);

    } else {
        m_i2ows.push_back (-1);
    }
    // get Ow from dict digitizer
    if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {

        if (-1 == DctSetId)
            Ow = m_w2ow_dct.Process (pText, TextLen);
        else
            Ow = m_w2ow_dct.Process (DctSetId);

        m_i2ows.push_back (Ow);

    } else {
        m_i2ows.push_back (-1);
    }

    m_WordCount++;
}


template < class Ty >
inline void FAGcInterpreter_t< Ty >::Prepare ()
{
    DebugLogAssert (1 <= m_WordCount && \
      m_i2ows.size () == (unsigned int) m_WordCount * FAFsmConst::DIGITIZER_COUNT);

    // add right anchor as a last word

    m_WordCount++;

    const int OldSize = m_i2ows.size ();
    m_i2ows.resize (OldSize + FAFsmConst::DIGITIZER_COUNT);

    int * pOut = m_i2ows.begin () + OldSize;
    for (int i = 0; i < FAFsmConst::DIGITIZER_COUNT; ++i) {
        pOut [i] = FAFsmConst::IW_R_ANCHOR;
    }

    m_pI2Ows = m_i2ows.begin ();
}


template < class Ty >
inline void FAGcInterpreter_t< Ty >::Clear ()
{
    m_WordCount = 1;
    m_i2ows.resize (FAFsmConst::DIGITIZER_COUNT);
}


template < class Ty >
inline const int FAGcInterpreter_t< Ty >::
    UpdateResults (const int From, const int To, const int State)
{
    DebugLogAssert (m_pLDB);
    DebugLogAssert (m_pActs);
    DebugLogAssert (0 <= From && 0 <= To && From <= To);

    // take left anchor into account
    int From2;
    if (0 < From) {
        From2 = From - 1;
        m_trbrs.SetBase (From - 1);
    } else {
        From2 = 0;
        m_trbrs.SetBase (-1);
    }
    // take right anchor into account
    int To2 = To - 1;
    if (To == (m_WordCount - 1)) {
        To2--;
    }

    const int Count = m_pRulesMap->GetOws (State, m_pRuleNums, m_MaxRuleCount);
    DebugLogAssert (0 < Count && Count <= m_MaxRuleCount);

    for (int i = 0; i < Count; ++i) {

        const int RuleNum = m_pRuleNums [i];

        // see whether sub-match is expected for this rule, NULL if not
        const FAWREConfCA * pR = m_pLDB->GetRule (RuleNum);

        if (pR) {
            // get the configuration
            const int TokenType = pR->GetTokenType ();
            const FARSDfaCA * pDfa1 = pR->GetDfa1 ();
            const FAMealyDfaCA * pSigma1 = pR->GetSigma1 ();
            const FARSDfaCA * pDfa2 = pR->GetDfa2 (); 
            const FAMealyDfaCA * pSigma2 = pR->GetSigma2 ();
            const FAMultiMapCA * pTrBrMap = pR->GetTrBrMap ();

            // set the configuration up
            m_r2trbr.SetTokenType (TokenType);
            m_r2trbr.SetMealy1 (pDfa1, pSigma1);
            m_r2trbr.SetMealy2 (pDfa2, pSigma2);
            m_r2trbr.SetTrBrMap (pTrBrMap);

            // calc input pointer and length
            const int * pIn = m_pI2Ows + (From * FAFsmConst::DIGITIZER_COUNT);
            DebugLogAssert ((unsigned int) From * FAFsmConst::DIGITIZER_COUNT < m_i2ows.size ());
            const int Size = (To - From + 1) * FAFsmConst::DIGITIZER_COUNT;
            DebugLogAssert (0 < Size && (unsigned int) Size <= m_i2ows.size ());

            // make a sub-match extraction
            m_r2trbr.Process2 (pIn, Size, &m_trbrs);
        }

        // call an action
        m_pActs->Process (RuleNum, From2, To2, &m_trbrs, m_pContext);

        m_trbrs.Clear ();
    }

    return From;
}


template < class Ty >
inline const int FAGcInterpreter_t< Ty >::
    UpdateResults_resolver (
            const int From,
            const int To,
            const int State
        )
{
    DebugLogAssert (m_pResolver);
    DebugLogAssert (0 <= From && 0 <= To && From <= To);

    const int Count = m_pRulesMap->GetOws (State, m_pRuleNums, m_MaxRuleCount);
    DebugLogAssert (0 < Count && Count <= m_MaxRuleCount);

    for (int i = 0; i < Count; ++i) {

        const int RuleNum = m_pRuleNums [i];
        m_pResolver->AddResult (RuleNum, From, To);
    }

    return From;
}


template < class Ty >
inline void FAGcInterpreter_t< Ty >::ApplyResults_resolver ()
{
    DebugLogAssert (m_pResolver && m_pActs);

    int RuleNum, From, To;

    m_pResolver->Process ();

    const int Count = m_pResolver->GetCount ();

    for (int i = 0; i < Count; ++i) {

        // get next resolved result
        m_pResolver->GetResult (i, &RuleNum, &From, &To);
        DebugLogAssert (0 <= From && 0 <= To && From <= To);

        // take left anchor into account
        int From2;
        if (0 < From) {
            From2 = From - 1;
            m_trbrs.SetBase (From - 1);
        } else {
            From2 = 0;
            m_trbrs.SetBase (-1);
        }
        // take right anchor into account
        int To2 = To - 1;
        if (To == (m_WordCount - 1)) {
            To2--;
        }

        // see whether sub-match is expected for this rule, NULL if not
        const FAWREConfCA * pR = m_pLDB->GetRule (RuleNum);

        if (pR) {
            // get the rest of configuration
            const int TokenType = pR->GetTokenType ();
            const FARSDfaCA * pDfa1 = pR->GetDfa1 ();
            const FAMealyDfaCA * pSigma1 = pR->GetSigma1 ();
            const FARSDfaCA * pDfa2 = pR->GetDfa2 (); 
            const FAMealyDfaCA * pSigma2 = pR->GetSigma2 ();
            const FAMultiMapCA * pTrBrMap = pR->GetTrBrMap ();

            // set the configuration up
            m_r2trbr.SetTokenType (TokenType);
            m_r2trbr.SetMealy1 (pDfa1, pSigma1);
            m_r2trbr.SetMealy2 (pDfa2, pSigma2);
            m_r2trbr.SetTrBrMap (pTrBrMap);

            // calc input pointer and length
            const int * pIn = m_pI2Ows + (From * FAFsmConst::DIGITIZER_COUNT);
            DebugLogAssert ((unsigned int) From * FAFsmConst::DIGITIZER_COUNT < m_i2ows.size ());
            const int Size = (To - From + 1) * FAFsmConst::DIGITIZER_COUNT;
            DebugLogAssert (0 < Size && (unsigned int) Size <= m_i2ows.size ());

            // make sub-match extraction
            m_r2trbr.Process2 (pIn, Size, &m_trbrs);
        }

        // call the action
        m_pActs->Process (RuleNum, From2, To2, &m_trbrs, m_pContext);

        m_trbrs.Clear ();
    }

    m_pResolver->Clear ();
}


template < class Ty >
inline void FAGcInterpreter_t< Ty >::Process_int ()
{
    int State, Iw;

    const int InitialState = m_pRulesDfa->GetInitial ();

    for (int j = 0; j < m_WordCount; ++j) {

        State = InitialState;

        const int FromPos = j;

        for (int i = FromPos; i < m_WordCount; ++i) {

            if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {

                Iw = m_pI2Ows [i * FAFsmConst::DIGITIZER_COUNT];
                State = m_pRulesDfa->GetDest (State, Iw);
                if (-1 == State)
                    break;
            }
            if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {

                Iw = m_pI2Ows [(i * FAFsmConst::DIGITIZER_COUNT) + 1];
                State = m_pRulesDfa->GetDest (State, Iw);
                if (-1 == State)
                    break;
            }
            if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {

                Iw = m_pI2Ows [(i * FAFsmConst::DIGITIZER_COUNT) + 2];
                State = m_pRulesDfa->GetDest (State, Iw);
                if (-1 == State)
                    break;
            }

            if (m_pRulesDfa->IsFinal (State)) {

                int Ji;

                if (m_pResolver) {
                    Ji = UpdateResults_resolver (FromPos, i, State);
                } else {
                    Ji = UpdateResults (FromPos, i, State);
                }

                if (Ji > j) {
                    j = Ji;
                }
            }
        } // of for (int i = 0; ...
    } // of for (int j = 0; ...

    if (m_pResolver) {
        ApplyResults_resolver ();
    }
}


template < class Ty >
void FAGcInterpreter_t< Ty >::Process ()
{
    Prepare ();
    Process_int ();
    Clear ();
}

}

#endif
