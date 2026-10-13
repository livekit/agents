/**
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */


#ifndef _FA_WORDGUESSER_T_H_
#define _FA_WORDGUESSER_T_H_

#include "FAConfig.h"
#include "FARSDfaCA.h"
#include "FAState2OwsCA.h"
#include "FAUtils_cl.h"
#include "FAFsmConst.h"
#include "FATransformCA_t.h"
#include "FAWgConfKeeper.h"
#include "FALimits.h"
#include "FASecurity.h"

namespace BlingFire
{

///
/// Word Guesser class
///

template < class Ty >
class FAWordGuesser_t {

public:
    FAWordGuesser_t ();
    virtual ~FAWordGuesser_t ();

public:
    /// Note: this method expects initialized configuration object
    /// pInTr can be NULL if no transformation assumed
    virtual void Initialize (
            const FAWgConfKeeper * pConf, 
            const FATransformCA_t < Ty > * pInTr
        );

    /// returns size of the output array or 0 if no guess was made
    /// returns -1 if error
    const int Process (
            const Ty * pWordStr, // input word
            const int WordLen,   // inpur word length
            const int ** ppTags  // output array of tags
        );

    /// returns size of the output array or 0 if no guess was made
    /// returns -1 if error
    /// Note: if the returned count larger than MaxTagCount then pTags values are undefined
    ///  it is up to caller code to handle this situation.
    const int Process (
            const Ty * pWordStr, // input word
            const int WordLen,   // inpur word length
            int * pTags,         // output array of tags allocated on client side
            const int MaxTagCount
        ) const;

private:
    /// returns the deepest state corresponding to the guess of the guesser
    /// returns -1, is none is found
    inline const int GetDeepestState (const Ty * pWordStr, const int WordLen) const;

    /// returns object into the initial state
    inline void Clear ();

private:
    const FARSDfaCA * m_pDfa;
    const FAState2OwsCA * m_pState2Ows;
    const FATransformCA_t < Ty > * m_pInTr;

    bool m_DictMode;
    bool m_IgnoreCase;
    int m_Direction;
    int m_MaxLen;
    int m_DefTag;

    int m_MaxOwCount;
    int * m_pOutOws;

    bool m_Ready;

    /// charmap
    const FAMultiMapCA * m_pCharMap;

    enum {
        DefMaxLen = FALimits::MaxWordLen,
        DefDelimIw = FAFsmConst::IW_ANY,
    };
};


template < class Ty >
FAWordGuesser_t< Ty >::FAWordGuesser_t () :
    m_pDfa (NULL),
    m_pState2Ows (NULL),
    m_pInTr (NULL),
    m_DictMode (false),
    m_IgnoreCase (false),
    m_Direction (FAFsmConst::DIR_L2R),
    m_MaxLen (DefMaxLen),
    m_DefTag (-1),
    m_MaxOwCount (0),
    m_pOutOws (NULL),
    m_Ready (false),
    m_pCharMap (NULL)
{
}


template < class Ty >
FAWordGuesser_t< Ty >::~FAWordGuesser_t ()
{
    FAWordGuesser_t< Ty >::Clear ();
}


template < class Ty >
inline void FAWordGuesser_t< Ty >::Clear ()
{
    m_DictMode = false;
    m_IgnoreCase = false;
    m_Direction = FAFsmConst::DIR_L2R;
    m_MaxLen = DefMaxLen;
    m_DefTag = -1;
    m_MaxOwCount = 0;
    m_Ready  = false;
    m_pCharMap = NULL;

    if (m_pOutOws) {
        delete [] m_pOutOws;
        m_pOutOws = NULL;
    }
}


template < class Ty >
void FAWordGuesser_t< Ty >::Initialize (
        const FAWgConfKeeper * pConf, 
        const FATransformCA_t < Ty > * pInTr
    )
{
    FAWordGuesser_t< Ty >::Clear ();

    DebugLogAssert (false == m_Ready);

    if (NULL == pConf) {
        return;
    }

    m_pDfa = pConf->GetRsDfa ();
    m_pState2Ows = pConf->GetState2Ows ();
    m_Direction = pConf->GetDirection ();
    m_DictMode = pConf->GetDictMode ();
    m_IgnoreCase = pConf->GetIgnoreCase ();
    m_DefTag = pConf->GetDefTag ();
    m_pCharMap = pConf->GetCharMap ();

    const bool NoTrUse = pConf->GetNoTrUse ();
    if (false == NoTrUse) {
        m_pInTr = pInTr;
    }

    const int MaxLen = pConf->GetMaxLen ();
    if (-1 != MaxLen) {
        m_MaxLen = MaxLen;
    }

    if (NULL != m_pState2Ows) {

        m_MaxOwCount = m_pState2Ows->GetMaxOwsCount ();

        if (0 < m_MaxOwCount) {
            DebugLogAssert (NULL == m_pOutOws);
            m_pOutOws = new int [m_MaxOwCount];
            LogAssert (m_pOutOws);
        }
    }

    m_Ready = m_pDfa && m_pState2Ows && 0 < m_MaxOwCount && m_pOutOws;

    LogAssert (m_Ready);
}


template < class Ty >
inline const int FAWordGuesser_t< Ty >::
    GetDeepestState (const Ty * pWordStr, const int WordSize) const
{
    DebugLogAssert (pWordStr && 0 < WordSize && FALimits::MaxWordLen >= WordSize);
    __analysis_assume (pWordStr && 0 < WordSize && FALimits::MaxWordLen >= WordSize);

    const int TmpBuffSize = 2 * FALimits::MaxWordLen;
    Ty TmpBuff [TmpBuffSize];

    const Ty * pIn = pWordStr;
    int InSize = WordSize;

    // normalize case, if needed
    if (m_IgnoreCase) {
        for (int i = 0 ; i < WordSize; ++i) {
            const int InSymbol = (int) pIn [i] ;
            const int OutSymbol = FAUtf32ToLower (InSymbol) ;
            TmpBuff [i] = (Ty) OutSymbol ;
        }
        pIn = TmpBuff;
    }
    // normalize characters
    if (m_pCharMap) {
        // in-place is fine
        InSize = FANormalizeWord (pIn, InSize, TmpBuff, TmpBuffSize, m_pCharMap);
        pIn = TmpBuff;
    }
    // apply transformation, if needed
    if (m_pInTr) {
        // apply the transformation, possibly in-place
        const int OutSize = m_pInTr->Process (pIn, InSize, TmpBuff, TmpBuffSize);
        // see whether there was some transformation made
        if (0 < OutSize && OutSize <= TmpBuffSize) {
            pIn = TmpBuff;
            InSize = OutSize;
        }
    }

    int State = m_pDfa->GetInitial ();
    int DstState = State;
    int Pos = 0;

    const int Size = (InSize > m_MaxLen) ? m_MaxLen : InSize ;

    if (false == m_DictMode) {

        // go as deep as possible
        if (FAFsmConst::DIR_R2L == m_Direction) {

            // read word right to left direction, take Size letters
            for (; Pos < Size; ++Pos) {
                const int Iw = (unsigned int) (pIn [InSize - Pos - 1]);
                DstState = m_pDfa->GetDest (State, Iw);
                if (-1 != DstState)
                    State = DstState;
                else
                    break;
            }
        } else {
            DebugLogAssert (FAFsmConst::DIR_L2R == m_Direction);

            // read word left to right direction, take Size letters
            for (; Pos < Size; ++Pos) {
                const int Iw = (unsigned int) (pIn [Pos]);
                DstState = m_pDfa->GetDest (State, Iw);
                if (-1 != DstState)
                    State = DstState;
                else
                    break;
            }
        }
        if (-1 != DstState) {
            // one more step for delimiter
            DebugLogAssert (State == DstState);
            DstState = m_pDfa->GetDest (State, DefDelimIw);
        }
        if (-1 != DstState) {
            State = DstState;
        }

    } else {

        if (FAFsmConst::DIR_L2R == m_Direction) {
            // read word left to right direction, take Size letters
            for (; Pos < Size && -1 != State; ++Pos) {
                const int Iw = (unsigned int) (pIn [Pos]);
                State = m_pDfa->GetDest (State, Iw);
            }
        } else {
            DebugLogAssert (FAFsmConst::DIR_R2L == m_Direction);
            // read word right to left direction, take Size letters
            for (; Pos < Size && -1 != State; ++Pos) {
                const int Iw = (unsigned int) (pIn [InSize - Pos - 1]);
                State = m_pDfa->GetDest (State, Iw);
            }
        }
        if (-1 != State) {
            // one more step for delimiter
            State = m_pDfa->GetDest (State, DefDelimIw);
        }
    } // of if (false == m_DictMode) ...

    return State;
}

template < class Ty >
const int FAWordGuesser_t< Ty >::
    Process (const Ty * pWordStr, const int WordSize, const int ** ppTags)
{
    DebugLogAssert (ppTags);

    if (!m_Ready) {
        return -1;
    }

    int State = 0;
    int OwsCount = 0;

    if (0 == WordSize) {
        goto exit_default;
    }

    // get a state corresponding to the input
    State = GetDeepestState (pWordStr, WordSize);
    if (-1 == State) {
        goto exit_default;
    }

    // get reaction of the deepest state
    OwsCount = m_pState2Ows->GetOws (State, m_pOutOws, m_MaxOwCount);
    if (0 < OwsCount) {
        *ppTags = m_pOutOws;
        return OwsCount;
    }

exit_default:
    if (-1 != m_DefTag) {
        *ppTags = & m_DefTag;
        return 1;
    } else {
        return -1;
    }
}


template < class Ty >
const int FAWordGuesser_t< Ty >::
    Process (const Ty * pWordStr, const int WordSize, int * pTags, const int MaxTagCount) const
{
    DebugLogAssert (pTags && 0 < MaxTagCount);

    if (!m_Ready) {
        return -1;
    }

    int State = 0;
    int OwsCount = 0;

    if (0 == WordSize) {
        goto exit_default;
    }

    // get a state corresponding to the input
    State = GetDeepestState (pWordStr, WordSize);
    if (-1 == State) {
        goto exit_default;
    }

    // get reaction of the deepest state
    OwsCount = m_pState2Ows->GetOws (State, pTags, MaxTagCount);
    if (0 < OwsCount) {
        return OwsCount;
    }

exit_default:
    if (-1 != m_DefTag) {
        pTags [0] = m_DefTag;
        return 1;
    } else {
        return -1;
    }
}

}

#endif
