/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_REGEXPTAGS_T_H_
#define _FA_REGEXPTAGS_T_H_

#include "FAConfig.h"
#include "FAFsmConst.h"
#include "FAUtf32Utils.h"
#include "FARSDfaCA.h"
#include "FAState2OwsCA.h"
#include "FAMultiMapCA.h"
#include "FAWbdConfKeeper.h"
#include "FALimits.h"

namespace BlingFire
{

///
/// For the given input string, this processor returns a set of tags with
/// scores. The tags and scores are calculated by matching fa_lex-style regular
/// expressions, unlike fa_lex interpreter this processor uses *all* matches.
///
/// Notes:
/// 1. see doc/design.txt for typical abbreviations
/// 2. The typical class usage is as follows:
///    1. one time SetConf
///    2. many times Process
///    3. SetConf, if needed
///    4. many times Process, again
///          ...
///

template < class Ty >
class FARegexpTags_t {

public:
    FARegexpTags_t ();
    ~FARegexpTags_t ();

public:
    /// sets up the data containers, NULL will just clear this object
    void Initialize (const FAWbdConfKeeper * pWbdConf);

    /// The input is a string, e.g. word, sentence, url, etc.
    /// The function returns size and two arrays the first keeps a set of tags
    /// the second keeps corresponding scores for each tag. 
    /// The results are valid only between Process calls.
    const int Process (
            const Ty * pIn,
            const int InSize,
            const int ** ppTags,
            const int ** ppScores
        );

private:
    inline void AddTags (const int FinalState);
    void Clear ();

private:
    /// input objects
    const FARSDfaCA * m_pDfa;
    const FAState2OwsCA * m_pState2Ows;
    const FAMultiMapCA * m_pActs;
    bool m_fIgnoreCase;
    /// temporary Ows buffer
    int * m_pOws;
    int m_MaxOwsCount;
    /// constants
    enum {
        MaxTokenLength = FALimits::MaxWordLen,
        MinActSize = 4,
        MaxActSize = 4,
        DefSubIw = FAFsmConst::IW_EPSILON,
    };
    /// runtime buffers, of a 0..MaxTag size
    /// Note: the memory is allocated ones at Initialize method
    int * m_pTag2Score;
    int * m_pTags;
    int * m_pScores;
    int m_MaxTagCount;
    int m_TagCount;
};


template < class Ty >
FARegexpTags_t< Ty >::FARegexpTags_t () :
    m_pDfa (NULL),
    m_pState2Ows (NULL),
    m_pActs (NULL),
    m_fIgnoreCase (false),
    m_pOws (NULL),
    m_MaxOwsCount (0),
    m_pTag2Score (NULL),
    m_pTags (NULL),
    m_pScores (NULL),
    m_MaxTagCount (0),
    m_TagCount (0)
{}


template < class Ty >
FARegexpTags_t< Ty >::~FARegexpTags_t ()
{
    FARegexpTags_t< Ty >::Clear ();
}


template < class Ty >
void FARegexpTags_t< Ty >::Clear ()
{
    if (m_pOws) {
        delete m_pOws;
        m_pOws = NULL;
    }
    if (m_pTag2Score) {
        delete m_pTag2Score;
        m_pTag2Score = NULL;
    }
    if (m_pTags) {
        delete m_pTags;
        m_pTags = NULL;
    }
    if (m_pScores) {
        delete m_pScores;
        m_pScores = NULL;
    }

    m_pDfa = NULL;
    m_pState2Ows = NULL;
    m_pActs = NULL;
    m_fIgnoreCase = false;
    m_MaxOwsCount = 0;
    m_MaxTagCount = 0;
    m_TagCount = 0;
}


template < class Ty >
void FARegexpTags_t< Ty >::Initialize (const FAWbdConfKeeper * pWbdConf)
{
    FARegexpTags_t< Ty >::Clear ();

    if (!pWbdConf) {
        return;
    }

    m_pDfa = pWbdConf->GetRsDfa ();
    m_pState2Ows = pWbdConf->GetState2Ows ();
    m_pActs = pWbdConf->GetMMap ();
    LogAssert (m_pDfa && m_pState2Ows && m_pActs);

    m_fIgnoreCase =  pWbdConf->GetIgnoreCase ();

    m_MaxOwsCount = m_pState2Ows->GetMaxOwsCount ();
    LogAssert (0 < m_MaxOwsCount);

    LogAssert (!m_pOws);
    m_pOws = new int [m_MaxOwsCount];
    LogAssert (m_pOws);

    m_MaxTagCount = 1 + pWbdConf->GetMaxTag ();
    LogAssert (0 < m_MaxTagCount);

    LogAssert (!m_pTag2Score);
    m_pTag2Score = new int [m_MaxTagCount];
    LogAssert (m_pTag2Score);

    LogAssert (!m_pTags);
    m_pTags = new int [m_MaxTagCount];
    LogAssert (m_pTags);

    LogAssert (!m_pScores);
    m_pScores = new int [m_MaxTagCount];
    LogAssert (m_pScores);

    for (int i = 0; i < m_MaxTagCount; ++i) {
        m_pTag2Score [i] = -1;
        m_pScores [i] = 0;
        m_pTags [i] = 0;
    }

    /// we need to validate all the actions so we may ignore all checks later:
    /// actions size, tag values, scores should be correct

    const int * pAct;
    int ActSize;
    int ActId = 0;

    while (-1 != (ActSize = m_pActs->Get (ActId++, &pAct))) {

        // invalid action
        LogAssert (pAct && MaxActSize >= ActSize && MinActSize <= ActSize);

        const int Score = pAct [2];
        const int Tag = pAct [3];

        // invalid score or tag value
        LogAssert (0 <= Score && 0 <= Tag && m_MaxTagCount > Tag);
    }
}


template < class Ty >
inline void FARegexpTags_t< Ty >::AddTags (const int FinalState)
{
    DebugLogAssert (m_pActs && m_pState2Ows);

    const int OwCount = m_pState2Ows->GetOws (FinalState, m_pOws, m_MaxOwsCount);
    DebugLogAssert (OwCount <= m_MaxOwsCount);

    for (int i = 0; i < OwCount; ++i) {

        const int Ow = m_pOws [i];

        const int * pAct;
#ifndef NDEBUG
        const int ActSize = 
#endif
            m_pActs->Get (Ow, &pAct);
        DebugLogAssert (MinActSize <= ActSize && pAct);

        const int Score = pAct [2];
        DebugLogAssert (0 <= Score);

        const int Tag = pAct [3];
        DebugLogAssert (0 <= Tag && m_MaxTagCount > Tag);

        const int OldScore = m_pTag2Score [Tag];

        // store the score
        if (OldScore < Score) {
            m_pTag2Score [Tag] = Score;
        }
        // add the tag if it was not there
        if (-1 == OldScore) {
            DebugLogAssert (m_TagCount < m_MaxTagCount);
            m_pTags [m_TagCount++] = Tag;
        }
    }
}


template < class Ty >
const int FARegexpTags_t< Ty >::
    Process (
            const Ty * pIn,
            const int InSize,
            const int ** ppTags,
            const int ** ppScores
        )
{
    LogAssert (m_pActs && m_pDfa && m_pState2Ows);
    LogAssert (ppTags && ppScores);

    int i, Iw, Dst;

    /// return scores into the initial state
    for (i = 0; i < m_TagCount; ++i) {

        const int Tag = m_pTags [i];
        DebugLogAssert (0 <= Tag && m_MaxTagCount > Tag);

        m_pTag2Score [Tag] = -1;
    }
    m_TagCount = 0;

    /// we keep track of the previous final state by optimization reasons
    int PrevFinal = -1;

    const int Initial = m_pDfa->GetInitial ();

    /// iterate thru all possible start positions
    for (int FromPos = -1; FromPos < InSize; ++FromPos) {

        int State = Initial;
        int j = FromPos;

        // maximum token length bounds j
        int LengthBound = FromPos + MaxTokenLength;
        if (InSize < LengthBound) {
            LengthBound = InSize;
        }

        /// feed the left anchor, if appropriate
        if (-1 == j) {
            State = m_pDfa->GetDest (Initial, FAFsmConst::IW_L_ANCHOR);
            if (-1 == State) {
                State = m_pDfa->GetDest (Initial, FAFsmConst::IW_ANY);
                if (-1 == State) {
                    continue;
                }
            }
            j++;
        }

        /// feed the letters
        for (; j < LengthBound; ++j) {

            Iw = pIn [j];
            // prevent regular input weights to match control input weights
            if (FAFsmConst::IW_EPSILON > Iw) {
                Iw = DefSubIw;
            }
            if (m_fIgnoreCase) {
                Iw = FAUtf32ToLower (Iw);
            }
            Dst = m_pDfa->GetDest (State, Iw);
            if (-1 == Dst) {
                Dst = m_pDfa->GetDest (State, FAFsmConst::IW_ANY);
                if (-1 == Dst) {
                    break;
                }
            }
            if (m_pDfa->IsFinal (Dst) && PrevFinal != Dst) {
                PrevFinal = Dst;
                AddTags (Dst);
            }
            State = Dst;
        } // of for (; j < InSize; ...

        /// feed the right anchor, if appropriate
        if (InSize == j) {

            DebugLogAssert (-1 != State);

            Dst = m_pDfa->GetDest (State, FAFsmConst::IW_R_ANCHOR);
            if (-1 == Dst) {
                Dst = m_pDfa->GetDest (State, FAFsmConst::IW_ANY);
            }
            if (-1 != Dst && PrevFinal != Dst && m_pDfa->IsFinal (Dst)) {
                PrevFinal = Dst;
                AddTags (Dst);
            }
        }

    } // of for (FromPos = -1; ...

    /// update the scores' array
    for (i = 0; i < m_TagCount; ++i) {

        const int Tag = m_pTags [i];
        DebugLogAssert (0 <= Tag && m_MaxTagCount > Tag);

        const int Score = m_pTag2Score [Tag];
        DebugLogAssert (0 <= Score);

        m_pScores [i] = Score;
    }

    *ppTags = m_pTags;
    *ppScores = m_pScores;
    return m_TagCount;
}

}

#endif
