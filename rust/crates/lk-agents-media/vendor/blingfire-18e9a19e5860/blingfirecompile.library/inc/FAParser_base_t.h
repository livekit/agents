/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_PARSER_BASE_T_H_
#define _FA_PARSER_BASE_T_H_

#include "FAConfig.h"
#include "FAArray_cont_t.h"
#include "FARSDfaCA.h"
#include "FAState2OwsCA.h"
#include "FAParseTreeA.h"
#include "FAFsmConst.h"
#include "FALimits.h"
#include "FADigitizer_t.h"
#include "FADigitizer_dct_t.h"

namespace BlingFire
{

class FAState2OwCA;
class FAAllocatorA;

///
/// This class keeps common functionality for Moore-WRE rules based parser.
/// (see FAParser2WRE.h for syntax description)
///

template < class Ty >
class FAParser_base_t {

public:
    FAParser_base_t (FAAllocatorA * pAlloc);
    virtual ~FAParser_base_t ();

public:
    /// sets up rules automaton
    virtual void SetRules (
            const FARSDfaCA * pRulesDfa,
            const FAState2OwsCA * pState2Rules
        );
    /// sets up RuleNum -> Action mapping
    virtual void SetActions (const FAMultiMapCA * pRule2Act);
    /// sets up text digitizer, if needed
    virtual void SetDigitizer (
            const FARSDfaCA * pDfa,
            const FAState2OwCA * pState2Ow
        );
    /// sets up dictionary digitizer, if needed
    virtual void SetDctDigitizer (
            const FADictInterpreter_t <Ty> * pTagDict,
            const FAArrayCA * pSetId2Ow
        );
    /// this flag is passed to the text digitizer
    virtual void SetIgnoreCase (const bool IgnoreCase);
    /// sets up WRE token type (a bitwise combination of WRE_TT_* constants)
    virtual void SetTokenType (const int TokenType);
    /// sets up base value for tag output weights
    virtual void SetTagOwBase (const int TagOwBase);
    /// sets up maximum amount of passes
    virtual void SetMaxPassCount (const int MaxPassCount);
    /// sets up output tree container
    virtual void SetParseTree (FAParseTreeA * pTree);
    /// sets up whether parser should resume from preexisting tree
    virtual void SetResume (const bool Resume);
    /// specifies a custom initial state
    virtual void SetCustomInitial (const int State);

public:
    /// adds next word, all options should already be setup
    void AddWord (
            const Ty * pText,       // word text
            const int TextLen,      // word text length
            const int Tag,          // POS tag (after disambiguation)
            const int DctSetId = -1 // tag dictionary id for the given word
        );
    /// makes parsing
    void Process ();

private:
    /// makes a lookup by a label at position i
    inline const int GetNextState (int State, const int i) const;
    /// finds all rule matches
    inline void FindResults ();

protected:
    /// makes parser ready to work 
    /// (called before parsing started)
    virtual void Prepare ();

    /// returns parser into initial state 
    /// (called after parsing stoped)
    virtual void Clear ();

    /// updates results storage and returns position to continue from,
    /// normally it should be the same position as FromPos
    /// (called several times per pass each time the final state is found)
    virtual const int UpdateResults (
            const int FromPos,
            const int ToPos,
            const int State
        ) = 0;

    /// applies all the results from a given pass to the tree
    /// returns true if parser should attempt one more pass
    /// or false to stop parsing
    /// (called each time current pass if finished)
    virtual const bool ApplyResults () = 0;

protected:
    /// rules automaton
    const FARSDfaCA * m_pRulesDfa;
    const FAState2OwsCA * m_pState2Rules;
    /// RuleNum -> Act mapping
    const FAMultiMapCA * m_pRule2Act;
    /// output tree
    FAParseTreeA * m_pTree;
    /// maximum amount of parsing iterations
    int m_MaxPassCount;
    /// tag Ow base
    int m_TagOwBase;
    /// WRE token type
    int m_TokenType;
    /// upper level count
    int m_UpperCount;
    /// upper labels
    const int * m_pLabels;
    /// resume flag
    bool m_Resume;
    /// custom initial state, -1 if not used
    int m_CustomInitial;

    /// text digitizer
    FADigitizer_t < Ty > m_w2ow;
    /// dict digitizer
    FADigitizer_dct_t < Ty > m_w2ow_dct;

    /// text digitizer output weights
    FAArray_cont_t < int > m_i2ow_txt;
    const int * m_pI2Ow_txt;
    /// tag "digitizer" output weights
    FAArray_cont_t < int > m_i2ow_tag;
    const int * m_pI2Ow_tag;
    /// dict "digitizer" output weights
    FAArray_cont_t < int > m_i2ow_dct;
    const int * m_pI2Ow_dct;

    /// rule nums storage
    FAArray_cont_t < int > m_Rules;
    int * m_pRules;
    int m_MaxRulesSize;


    enum {
        DefMaxPassCount = 1,
    };
};


template < class Ty >
FAParser_base_t< Ty >::FAParser_base_t (FAAllocatorA * pAlloc) :
    m_pRulesDfa (NULL),
    m_pState2Rules (NULL),
    m_pRule2Act (NULL),
    m_pTree (NULL),
    m_MaxPassCount (DefMaxPassCount),
    m_TagOwBase (FAFsmConst::IW_EPSILON + 1),
    m_TokenType (FAFsmConst::WRE_TT_DEFAULT),
    m_UpperCount (0),
    m_pLabels (NULL),
    m_Resume (false),
    m_CustomInitial (-1),
    m_pI2Ow_txt (NULL),
    m_pI2Ow_tag (NULL),
    m_pI2Ow_dct (NULL),
    m_pRules (NULL),
    m_MaxRulesSize (0)
{
    m_w2ow.SetAnyIw (FAFsmConst::IW_ANY);
    m_w2ow.SetAnyOw (FAFsmConst::IW_ANY);

    m_w2ow_dct.SetAnyOw (FAFsmConst::IW_ANY);

    m_i2ow_txt.SetAllocator (pAlloc);
    m_i2ow_txt.Create ();

    m_i2ow_tag.SetAllocator (pAlloc);
    m_i2ow_tag.Create ();

    m_i2ow_dct.SetAllocator (pAlloc);
    m_i2ow_dct.Create ();

    m_Rules.SetAllocator (pAlloc);
    m_Rules.Create ();
}


template < class Ty >
FAParser_base_t< Ty >::~FAParser_base_t ()
{}


template < class Ty >
void FAParser_base_t< Ty >::
    SetRules (
        const FARSDfaCA * pRulesDfa,
        const FAState2OwsCA * pState2Rules
    )
{
    m_pRulesDfa = pRulesDfa;
    m_pState2Rules = pState2Rules;

    if (m_pState2Rules) {

        m_MaxRulesSize = m_pState2Rules->GetMaxOwsCount ();
        m_Rules.resize (m_MaxRulesSize);
        m_pRules = m_Rules.begin ();
    }
}


template < class Ty >
void FAParser_base_t< Ty >::SetActions (const FAMultiMapCA * pRule2Act)
{
    m_pRule2Act = pRule2Act;
}


template < class Ty >
void FAParser_base_t< Ty >::SetDigitizer (
        const FARSDfaCA * pDfa,
        const FAState2OwCA * pState2Ow
    )
{
    m_w2ow.SetRsDfa (pDfa);
    m_w2ow.SetState2Ow (pState2Ow);

    if (pDfa && pState2Ow) {
        m_w2ow.Prepare ();
    }
}


template < class Ty >
void FAParser_base_t< Ty >::
    SetDctDigitizer (
        const FADictInterpreter_t <Ty> * pTagDict,
        const FAArrayCA * pSetId2Ow
    )
{
    m_w2ow_dct.SetTagDict (pTagDict);
    m_w2ow_dct.SetSet2Ow (pSetId2Ow);
}


template < class Ty >
void FAParser_base_t< Ty >::SetIgnoreCase (const bool IgnoreCase)
{
    m_w2ow.SetIgnoreCase (IgnoreCase);
}


template < class Ty >
void FAParser_base_t< Ty >::SetTokenType (const int TokenType)
{
    DebugLogAssert (TokenType & FAFsmConst::WRE_TT_TEXT || \
            TokenType & FAFsmConst::WRE_TT_TAGS || \
            TokenType & FAFsmConst::WRE_TT_DCTS);

    m_TokenType = TokenType;
}


template < class Ty >
void FAParser_base_t< Ty >::SetTagOwBase (const int TagOwBase)
{
    DebugLogAssert (0 <= TagOwBase);
    m_TagOwBase = TagOwBase;
}


template < class Ty >
void FAParser_base_t< Ty >::SetMaxPassCount (const int MaxPassCount)
{
    DebugLogAssert (0 < MaxPassCount);
    m_MaxPassCount = MaxPassCount;
}


template < class Ty >
void FAParser_base_t< Ty >::SetParseTree (FAParseTreeA * pTree)
{
    DebugLogAssert (pTree);
    m_pTree = pTree;
}


template < class Ty >
void FAParser_base_t< Ty >::SetResume (const bool Resume)
{
    m_Resume = Resume;
}


template < class Ty >
void FAParser_base_t< Ty >::SetCustomInitial (const int State)
{
    m_CustomInitial = State;
}


template < class Ty >
void FAParser_base_t< Ty >::AddWord (
        const Ty * pText, 
        const int TextLen, 
        const int Tag,
        const int DctSetId
    )
{
    DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);

    int Ow;

    if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {

        DebugLogAssert (pText && 0 < TextLen);
        Ow = m_w2ow.Process (pText, TextLen);
        m_i2ow_txt.push_back (Ow);
    }
    if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {

        DebugLogAssert (0 < Tag);
        Ow = Tag + m_TagOwBase;
        m_i2ow_tag.push_back (Ow);
    }
    if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {

        DebugLogAssert (pText && 0 < TextLen);
        if (-1 == DctSetId) {
            Ow = m_w2ow_dct.Process (pText, TextLen);
        } else {
            Ow = m_w2ow_dct.Process (DctSetId);
        }
        m_i2ow_dct.push_back (Ow);
    }

    m_UpperCount++;
}


template < class Ty >
inline const int FAParser_base_t< Ty >::
    GetNextState (int State, const int i) const
{
    DebugLogAssert (m_pLabels && m_pRulesDfa && \
            m_pI2Ow_txt && m_pI2Ow_tag && m_pI2Ow_dct);

    int Iw;

    // this is normal symbol
    if (0 <= i && i < m_UpperCount) {

        const int L = m_pLabels [i];

        if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {

            if (0 <= L) {
                DebugLogAssert ((unsigned int) L < m_i2ow_txt.size ());
                Iw = m_pI2Ow_txt [L];
            } else {
                Iw = FAFsmConst::IW_ANY;
            }
            State = m_pRulesDfa->GetDest (State, Iw);
        }
        if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {

            if (0 <= L) {
                DebugLogAssert ((unsigned int) L < m_i2ow_tag.size ());
                Iw = m_pI2Ow_tag [L];
            } else {
                Iw = m_TagOwBase - L;
            }
            State = m_pRulesDfa->GetDest (State, Iw);
        }
        if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {

            if (0 <= L) {
                DebugLogAssert ((unsigned int) L < m_i2ow_dct.size ());
                Iw = m_pI2Ow_dct [L];
            } else {
                Iw = FAFsmConst::IW_ANY;
            }
            State = m_pRulesDfa->GetDest (State, Iw);
        }

    // process the left anchor
    } else if (-1 == i) {
        if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {
          State = m_pRulesDfa->GetDest (State, FAFsmConst::IW_L_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {
          State = m_pRulesDfa->GetDest (State, FAFsmConst::IW_L_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {
          State = m_pRulesDfa->GetDest (State, FAFsmConst::IW_L_ANCHOR);
        }

    // process the right anchor
    } else {
        DebugLogAssert (m_UpperCount == i);
        if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {
          State = m_pRulesDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {
          State = m_pRulesDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {
          State = m_pRulesDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
        }
    }

    return State;
}


template < class Ty >
inline void FAParser_base_t< Ty >::FindResults ()
{
    DebugLogAssert (m_pRulesDfa);

    int State;

    // update input labels
    m_UpperCount = m_pTree->GetUpperLabels (&m_pLabels);

    // save the initial state
    const int InitialState = -1 == m_CustomInitial ? \
        m_pRulesDfa->GetInitial () : m_CustomInitial;

    // -1 to include the left anchor
    for (int j = -1; j < m_UpperCount; ++j) {

        State = InitialState;

        const int FromPos = j;

        // <= to include the right anchor
        for (int i = FromPos; i <= m_UpperCount; ++i) {

            // lookup the next state
            State = GetNextState (State, i);

            if (-1 == State)
                break;

            if (m_pRulesDfa->IsFinal (State)) {

                const int Ji = UpdateResults (FromPos, i, State);

                if (Ji > j) {
                    j = Ji;
                }
            }

        } // of for (int i = 0; ...

    } // of for (int j = 0; ...
}


template < class Ty >
void FAParser_base_t< Ty >::Clear ()
{
    m_i2ow_txt.resize (0);
    m_i2ow_tag.resize (0);
    m_i2ow_dct.resize (0);

    m_UpperCount = 0;
    m_pLabels = NULL;
}


template < class Ty >
void FAParser_base_t< Ty >::Prepare ()
{
    // update fast access pointers
    m_pI2Ow_txt = m_i2ow_txt.begin ();
    m_pI2Ow_tag = m_i2ow_tag.begin ();
    m_pI2Ow_dct = m_i2ow_dct.begin ();

    // create an empty tree, if needed
    if (!m_Resume) {
        m_pTree->Init (m_UpperCount);
    }
}


template < class Ty >
void FAParser_base_t< Ty >::Process ()
{
    Prepare ();

    bool Continue = true;
    int PassCount = 0;

    while (Continue && ++PassCount <= m_MaxPassCount) {

        FindResults ();

        Continue = ApplyResults ();

        m_pTree->Update ();
    }

    Clear ();
}

}

#endif
