/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WRELEXTOOLS_T_H_
#define _FA_WRELEXTOOLS_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAWREConfCA.h"
#include "FAMultiMapCA.h"
#include "FAState2OwsCA.h"
#include "FARSDfaCA.h"
#include "FAParserConfKeeper.h"
#include "FALimits.h"
#include "FASecurity.h"
#include "FAParseTreeA.h"
#include "FADigitizer_t.h"
#include "FADigitizer_dct_t.h"
#include "FAArray_cont_t.h"

namespace BlingFire
{

///
/// WRE based parser runtime, allows to do bottom-up and top-down parsing.
/// 
/// This runtime interprets the rules of the following format:
///
/// < wre > wre* --> Tag
/// < wre > wre* --> _call FnTag
/// < wre > wre* --> Tag _call FnTag
/// < wre > wre* --> _call FnTag1 FnTag2 ...
/// < wre > wre* --> Tag _call FnTag1 FnTag2 ...
///
/// _function FnTag
///  <same kind of rules>
/// _end
///
/// Notes:
///
///   1. The notation is similar to the fa_lex rules, see FALexTools_t.h, but
///      uses WRE for matching see doc\wre-tutorial.htm and 
///      doc\NeExtractionWithWre.ppt.
///
///   2. The rules should not have left context.
/// 

template < class Ty >
class FAWreLexTools_t {

public:
    FAWreLexTools_t ();
    ~FAWreLexTools_t ();

public:
    /// Sets up the data containers, can be called more than once
    /// If called for not the first time then pMemMgr can be NULL, in this case 
    ///  a previous memory manager is used.
    void Initialize (
            FAAllocatorA * pMemMgr,
            const FAParserConfKeeper * pParserConf, 
            const FADictInterpreter_t < Ty > * pDict = NULL
        );

    /// sets up maximum word count
    void Reset (const int MaxWordCount);

    /// adds next word, all options should already be setup
    /// adds up to MaxWordCount words, other words are ignored
    void AddWord (
            const Ty * pText,       // word text
            const int TextLen,      // word text length
            const int Tag,          // POS tag (after disambiguation)
            const int DctSetId = -1 // tag dictionary id for the given word
        );

    /// sets up initialized parse tree
    void SetParseTree (FAParseTreeA * pTree);

    /// makes a processing, returns a total amount of new constituents
    /// added to the parse tree
    const int Process ();

private:
    /// returns destination state from State by the i-th word
    inline const int GetNextState (int State, const int i) const;

    // makes analysis for the case when MaxLeftContext is 0
    const int Process_0_n (
            const int Initial,
            const int Offset,
            const int InSize,
            const int RecDepth,
            const bool fOnce = false
        );

private:
    /// rules automaton
    const FARSDfaCA * m_pDfa;
    const FAState2OwsCA * m_pState2Ows;
    /// RuleNum -> Act mapping
    const FAMultiMapCA * m_pActs;
    /// tag Ow base
    int m_TagOwBase;
    /// WRE token type
    int m_TokenType;

    /// text digitizer
    FADigitizer_t < Ty > m_w2ow;
    /// dict digitizer
    FADigitizer_dct_t < Ty > m_w2ow_dct;

    int m_MaxDepth;
    int m_MaxPassCount;

    /// maps function id into an initial state, or -1 if not valid
    const int * m_pFn2Ini;
    unsigned int m_Fn2IniSize;

    /// constants
    enum {
        DefMaxDepth = 2,
        DefMaxPassCount = 1,
        MinActSize = 3,
        DefSubIw = FAFsmConst::IW_EPSILON,
    };

    /// TODO: remove this when switched to an ordinary Moore FSM
    int * m_pRules;
    int m_MaxRuleCount;

    int m_WordCount;

    /// text digitizer output weights
    int * m_pI2Ow_txt;
    /// tag "digitizer" output weights
    int * m_pI2Ow_tag;
    /// dict "digitizer" output weights
    int * m_pI2Ow_dct;
    /// length of m_pI2Ow_* arrays, if not NULL
    int m_MaxWordCount;

    /// a buffer to keep digitizers' output-weights (ows)
    FAArray_cont_t < int > m_ows;
    /// an array to keep all matched rule ids
    FAArray_cont_t < int > m_rules;

    /// keeps track of the To of the last added constituent
    int m_LastTo;
    /// parse tree
    FAParseTreeA * m_pTree;
    /// upper level count
    int m_UpperCount;
    /// upper labels
    const int * m_pLabels;

    /// memory manager
    FAAllocatorA * m_pMemMgr;

};


template < class Ty >
FAWreLexTools_t< Ty >::FAWreLexTools_t () :
    m_pDfa (NULL),
    m_pState2Ows (NULL),
    m_pActs (NULL),
    m_TagOwBase (FAFsmConst::IW_EPSILON + 1),
    m_TokenType (FAFsmConst::WRE_TT_DEFAULT),
    m_MaxDepth (DefMaxDepth),
    m_MaxPassCount (DefMaxPassCount),
    m_pFn2Ini (NULL),
    m_Fn2IniSize (0),
    m_pRules (NULL),
    m_MaxRuleCount (0),
    m_WordCount (0),
    m_pI2Ow_txt (NULL),
    m_pI2Ow_tag (NULL),
    m_pI2Ow_dct (NULL),
    m_MaxWordCount (0),
    m_LastTo (0),
    m_pTree (NULL),
    m_UpperCount (0),
    m_pLabels (NULL),
    m_pMemMgr (NULL)
{}


template < class Ty >
FAWreLexTools_t< Ty >::~FAWreLexTools_t ()
{}


template < class Ty >
void FAWreLexTools_t< Ty >::SetParseTree (FAParseTreeA * pTree)
{
    m_pTree = pTree;
}


template < class Ty >
void FAWreLexTools_t< Ty >::
    Initialize (
        FAAllocatorA * pMemMgr, 
        const FAParserConfKeeper * pParserConf, 
        const FADictInterpreter_t < Ty > * pDict
    )
{
    // see if memory reinitialization is not required
    if (NULL != pMemMgr)
    {
        m_pMemMgr = pMemMgr;
        m_rules.SetAllocator (pMemMgr);
        m_rules.Clear ();
        m_ows.SetAllocator (pMemMgr);
        m_ows.Clear ();
    }

    // no allocator is set
    LogAssert (m_pMemMgr);

    // initialize from the model data
    if (pParserConf) {

        // get max depth and max pass count
        m_MaxDepth = pParserConf->GetMaxDepth ();
        m_MaxPassCount = pParserConf->GetMaxPassCount ();

        const FAWREConfCA * pWre = pParserConf->GetWre ();
        LogAssert (pWre);

        // get token type mask
        m_TokenType = pWre->GetTokenType ();
        // get tag Ow base
        m_TagOwBase = pWre->GetTagOwBase ();

        // setup rules automaton
        m_pDfa = pWre->GetDfa1 ();
        m_pState2Ows = pWre->GetState2Ows ();
        LogAssert (m_pDfa && m_pState2Ows);

        // allocate place to store alternative rules
        m_MaxRuleCount = m_pState2Ows->GetMaxOwsCount ();
        m_rules.Clear ();
        m_rules.resize (m_MaxRuleCount);
        m_pRules = m_rules.begin ();
        LogAssert (m_pRules);

        // setup text digitizer
        const FARSDfaCA * pDfa = pWre->GetTxtDigDfa ();
        const FAState2OwCA * pState2Ow = pWre->GetTxtDigOws ();
        const bool IgnoreCase = pParserConf->GetIgnoreCase ();

        LogAssert (!(FAFsmConst::WRE_TT_TEXT & m_TokenType) || 
                    (pDfa && pState2Ow));

        if (pDfa && pState2Ow) {
            m_w2ow.SetRsDfa (pDfa);
            m_w2ow.SetState2Ow (pState2Ow);
            m_w2ow.SetIgnoreCase (IgnoreCase);
            m_w2ow.Prepare ();
        }

        // setup dictionary digitizer
        const FAArrayCA * pSet2Ow = pWre->GetDictDig ();

        LogAssert (!(FAFsmConst::WRE_TT_DCTS & m_TokenType) || 
                    (pDict && pSet2Ow));

        if (pSet2Ow) {
            m_w2ow_dct.SetTagDict (pDict);
            m_w2ow_dct.SetSet2Ow (pSet2Ow);
        }

        // setup actions
        m_pActs = pParserConf->GetActs ();

        // setup function initial states
        m_Fn2IniSize = pParserConf->GetFnIniStates (&m_pFn2Ini);

    } else {

        m_pDfa = NULL;
        m_pState2Ows = NULL;
        m_pActs = NULL;
        m_pFn2Ini = NULL;
    }
}


template < class Ty >
void FAWreLexTools_t< Ty >::Reset (const int MaxWordCount)
{
    LogAssert (0 <= MaxWordCount);

    int TupleSize = 0;
    if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {
        TupleSize++;
    }
    if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {
        TupleSize++;
    }
    if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {
        TupleSize++;
    }

    m_ows.Clear ();
    m_ows.resize (MaxWordCount * TupleSize);
    int * pOws = m_ows.begin ();
    LogAssert (pOws);

    if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {
        m_pI2Ow_txt = pOws;
        pOws += MaxWordCount;
    }
    if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {
        m_pI2Ow_tag = pOws;
        pOws += MaxWordCount;
    }
    if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {
        m_pI2Ow_dct = pOws;
        pOws += MaxWordCount;
    }

    m_WordCount = 0;
    m_MaxWordCount = MaxWordCount;
}


template < class Ty >
void FAWreLexTools_t< Ty >::AddWord (
    const Ty * pText, const int TextLen, const int Tag, const int DctSetId)
{
    // Reset(WordCount) was called with smaller amount of words
    LogAssert (m_WordCount < m_MaxWordCount);

    if (FAFsmConst::WRE_TT_TEXT & m_TokenType) {
        const int Ow = m_w2ow.Process (pText, TextLen);
        m_pI2Ow_txt [m_WordCount] = Ow;
    }
    if (FAFsmConst::WRE_TT_TAGS & m_TokenType) {
        DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);
        const int Ow = Tag + m_TagOwBase;
        m_pI2Ow_tag [m_WordCount] = Ow;
    }
    if (FAFsmConst::WRE_TT_DCTS & m_TokenType) {
        int Ow;
        if (-1 == DctSetId) {
            Ow = m_w2ow_dct.Process (pText, TextLen);
        } else {
            Ow = m_w2ow_dct.Process (DctSetId);
        }
        m_pI2Ow_dct [m_WordCount] = Ow;
    }

    m_WordCount++;
}


template < class Ty >
inline const int FAWreLexTools_t< Ty >::
    GetNextState (int State, const int i) const
{
    // this is a normal word or node index
    if (0 <= i && i < m_UpperCount) {

        const int L = m_pLabels [i];

        // Is L is a constituent?
        if (0 > L) {
            if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {
                State = m_pDfa->GetDest (State, FAFsmConst::IW_ANY);
            }
            if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {
                State = m_pDfa->GetDest (State, -L + m_TagOwBase);
            }
            if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {
                State = m_pDfa->GetDest (State, FAFsmConst::IW_ANY);
            }
        // else L is an index of a word
        } else {
            if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {
                DebugLogAssert (L < m_WordCount);
                State = m_pDfa->GetDest (State, m_pI2Ow_txt [L]);
            }
            if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {
                DebugLogAssert (L < m_WordCount);
                State = m_pDfa->GetDest (State, m_pI2Ow_tag [L]);
            }
            if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {
                DebugLogAssert (L < m_WordCount);
                State = m_pDfa->GetDest (State, m_pI2Ow_dct [L]);
            }
        }

    // process the left anchor
    } else if (-1 == i) {
        if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {
          State = m_pDfa->GetDest (State, FAFsmConst::IW_L_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {
          State = m_pDfa->GetDest (State, FAFsmConst::IW_L_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {
          State = m_pDfa->GetDest (State, FAFsmConst::IW_L_ANCHOR);
        }

    // process the right anchor
    } else {
        DebugLogAssert (m_UpperCount == i);
        if (FAFsmConst::WRE_TT_TEXT & m_TokenType && -1 != State) {
          State = m_pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_TAGS & m_TokenType && -1 != State) {
          State = m_pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
        }
        if (FAFsmConst::WRE_TT_DCTS & m_TokenType && -1 != State) {
          State = m_pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
        }
    }

    return State;
}


template < class Ty >
const int FAWreLexTools_t< Ty >::
    Process_0_n (
            const int Initial,
            const int Offset,
            const int InSize,
            const int RecDepth,
            const bool fOnce
        )
{
    int OutSize = 0;
    int Dst;

    if (m_MaxDepth < RecDepth) {
        return 0;
    }

    /// iterate thru all possible start positions
    for (int FromPos = -1; FromPos < InSize; ++FromPos) {

        int State = Initial;
        int FinalState = -1;
        int FinalPos = -1;

        int j = FromPos;

        /// feed the left anchor, if appropriate
        if (-1 == j) {
            State = GetNextState (Initial, -1);
            if (-1 == State) {
                continue;
            }
            j++;
        }

        /// feed the InSize words up from the position Offset
        for (; j < InSize; ++j) {

            Dst = GetNextState (State, j + Offset);
            if (-1 == Dst) {
                break;
            }
            if (m_pDfa->IsFinal (Dst)) {
                FinalState = Dst;
                FinalPos = j;
            }
            State = Dst;

        } // of for (; j < InSize; ...

        /// feed the right anchor, if appropriate
        if (InSize == j) {
            DebugLogAssert (-1 != State);
            Dst = GetNextState (State, m_UpperCount);
            if (-1 != Dst && m_pDfa->IsFinal (Dst)) {
                FinalState = Dst;
                FinalPos = j;
            }
        }

        // use the FinalState and the deepest FinalPos(ition)
        if (-1 != FinalPos) {
            DebugLogAssert (-1 != FinalState);
            DebugLogAssert (FinalPos >= FromPos);

            /// TODO: switch to an ordinary Moore FSM, e.g. use FAState2OwCA
            const int OwCount = m_pState2Ows->GetOws (FinalState, m_pRules, m_MaxRuleCount);
            LogAssert (0 < OwCount);
            const int Ow = m_pRules [0];

            const int * pAct;
            const int ActSize = m_pActs->Get (Ow, &pAct);
            DebugLogAssert (MinActSize <= ActSize && pAct);

            const int LeftCx = pAct [0];
            const int RightCx = pAct [1];
            const int Tag = pAct [2];

            // From position in the data 
            int FromPos2 = FromPos + LeftCx;
            if (0 > FromPos2)
            {
                FromPos2 = 0;
            }
            else if (InSize <= FromPos2)
            {
                FromPos2 = InSize - 1;
            }

            // To position in the data 
            int ToPos2 = FinalPos - RightCx;
            if (0 > ToPos2)
            {
                ToPos2 = 0;
            }
            else if (InSize <= ToPos2)
            {
                ToPos2 = InSize - 1;
            }

            int FnIdx = MinActSize;

            // create the token, if Tag is specified
            if (0 != Tag) {
                DebugLogAssert (FALimits::MinTag <= Tag && FALimits::MaxTag >= Tag);
                // add a constituent, all constituent lables are negative
                m_pTree->AddNode (-Tag, FromPos2 + Offset, ToPos2 + Offset);
                // remember last To position, for the case when a sequence of functions is called
                m_LastTo = ToPos2 + Offset;
                OutSize++;
                FnIdx = MinActSize + 1;
            }

            // set "once" flag for called functions, if there is more than one
            const bool fFnOnce = 1 < (ActSize - FnIdx);
            // functions' starting position
            int FnFrom = FromPos2;

            // apply functions, if any
            for (; FnIdx < ActSize; ++FnIdx) {

                const int FnId = pAct [FnIdx];
                DebugLogAssert (0 <= FnId && (unsigned) FnId < m_Fn2IniSize);

                const int FnIni = m_pFn2Ini [FnId];
                DebugLogAssert (-1 != FnIni);

                const int FnInSize = ToPos2 - FnFrom + 1;

                const int FnOutSize = Process_0_n (FnIni, FnFrom + Offset, \
                  FnInSize, RecDepth + 1, 0 == FnId ? false : fFnOnce);

                // see if any tokens have been extracted
                if (0 < FnOutSize) {
                    // updated the total output count
                    OutSize += FnOutSize;
                    // next function starts from the last token's To + 1
                    FnFrom = m_LastTo + 1 - Offset;
                    // see if no text is left
                    if (FnFrom > ToPos2) {
                        break;
                    }
                }
            } // of for (; FnIdx < ActSize; ...

            // check if the function is supposed to be called once
            if (fOnce) {
                return OutSize;
            }

            // see if we can move the FromPos to the right more than one step
            if (FinalPos - RightCx > FromPos)
            {
                FromPos = FinalPos - RightCx;
            }

        } // of if (-1 != FinalPos)

    } // of for (FromPos = 0;

    return OutSize;
}


template < class Ty >
const int FAWreLexTools_t< Ty >::Process ()
{
    if (!m_pActs || !m_pDfa || !m_pState2Ows) {
        return -1;
    }

    int OutSize = 0;

    // do up to m_MaxPassCount passes
    for (int i = 0; i < m_MaxPassCount; ++i) {

        // get the initial state of the rules' automaton
        const int Initial = m_pDfa->GetInitial ();

        // update input labels
        m_UpperCount = m_pTree->GetUpperLabels (&m_pLabels);

        // do the pattern matching
        const int CurrOutSize = Process_0_n (Initial, 0, m_UpperCount, 1);

        // the tree has not been modified
        if (0 == CurrOutSize) {
            break;
        }

        // update the total amount of constituents added
        OutSize += CurrOutSize;

        // make sure the tree is ready
        m_pTree->Update ();
    }

    return OutSize;
}

}

#endif
